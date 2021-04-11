#include "ResidualGlobalCorrectionMakerBase.h"

// required for Transient Tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
// required for vtx fitting
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/MultiTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/MultiTrackPointingKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/CombinedKinematicConstraint.h"

#include "Math/Vector4Dfwd.h"


class ResidualGlobalCorrectionMakerTwoTrack : public ResidualGlobalCorrectionMakerBase
{
public:
  explicit ResidualGlobalCorrectionMakerTwoTrack(const edm::ParameterSet &);
  ~ResidualGlobalCorrectionMakerTwoTrack() {}

//   static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  
  virtual void beginStream(edm::StreamID) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  
  bool doMassConstraint_;
  double massConstraint_;
  
  
  float Jpsi_x;
  float Jpsi_y;
  float Jpsi_z;
  float Jpsi_pt;
  float Jpsi_eta;
  float Jpsi_phi;
  float Jpsi_mass;
  
  float Muplus_pt;
  float Muplus_eta;
  float Muplus_phi;
  
  float Muminus_pt;
  float Muminus_eta;
  float Muminus_phi;
  
  float Jpsigen_x;
  float Jpsigen_y;
  float Jpsigen_z;
  float Jpsigen_pt;
  float Jpsigen_eta;
  float Jpsigen_phi;
  float Jpsigen_mass;
  
  float Muplusgen_pt;
  float Muplusgen_eta;
  float Muplusgen_phi;
  
  float Muminusgen_pt;
  float Muminusgen_eta;
  float Muminusgen_phi;
  
  std::array<float, 5> Muplus_refParms;
  std::array<float, 5> MuMinus_refParms;
  
  std::vector<float> Muplus_jacRef;
  std::vector<float> Muminus_jacRef;

  
};


ResidualGlobalCorrectionMakerTwoTrack::ResidualGlobalCorrectionMakerTwoTrack(const edm::ParameterSet &iConfig) : ResidualGlobalCorrectionMakerBase(iConfig) 
{
  doMassConstraint_ = iConfig.getParameter<bool>("doMassConstraint");
  massConstraint_ = iConfig.getParameter<double>("massConstraint");
}

void ResidualGlobalCorrectionMakerTwoTrack::beginStream(edm::StreamID streamid)
{
  ResidualGlobalCorrectionMakerBase::beginStream(streamid);
  
  if (fillTrackTree_) {
    tree->Branch("Jpsi_x", &Jpsi_x);
    tree->Branch("Jpsi_y", &Jpsi_y);
    tree->Branch("Jpsi_z", &Jpsi_z);
    tree->Branch("Jpsi_pt", &Jpsi_pt);
    tree->Branch("Jpsi_eta", &Jpsi_eta);
    tree->Branch("Jpsi_phi", &Jpsi_phi);
    tree->Branch("Jpsi_mass", &Jpsi_mass);
    
    tree->Branch("Muplus_pt", &Muplus_pt);
    tree->Branch("Muplus_eta", &Muplus_eta);
    tree->Branch("Muplus_phi", &Muplus_phi);
    
    tree->Branch("Muminus_pt", &Muminus_pt);
    tree->Branch("Muminus_eta", &Muminus_eta);
    tree->Branch("Muminus_phi", &Muminus_phi);
    
    tree->Branch("Jpsigen_x", &Jpsigen_x);
    tree->Branch("Jpsigen_y", &Jpsigen_y);
    tree->Branch("Jpsigen_z", &Jpsigen_z);
    tree->Branch("Jpsigen_pt", &Jpsigen_pt);
    tree->Branch("Jpsigen_eta", &Jpsigen_eta);
    tree->Branch("Jpsigen_phi", &Jpsigen_phi);
    tree->Branch("Jpsigen_mass", &Jpsigen_mass);
    
    tree->Branch("Muplusgen_pt", &Muplusgen_pt);
    tree->Branch("Muplusgen_eta", &Muplusgen_eta);
    tree->Branch("Muplusgen_phi", &Muplusgen_phi);
    
    tree->Branch("Muminusgen_pt", &Muminusgen_pt);
    tree->Branch("Muminusgen_eta", &Muminusgen_eta);
    tree->Branch("Muminusgen_phi", &Muminusgen_phi);
    
    tree->Branch("Muplus_refParms", Muplus_refParms.data(), "Muplus_refParms[5]/F");
    tree->Branch("MuMinus_refParms", MuMinus_refParms.data(), "MuMinus_refParms[5]/F");
    
    tree->Branch("Muplus_jacRef", &Muplus_jacRef);
    tree->Branch("Muminus_jacRef", &Muminus_jacRef);
    
  }
}


// ------------ method called for each event  ------------
void ResidualGlobalCorrectionMakerTwoTrack::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  
  const bool dogen = fitFromGenParms_;
  
  using namespace edm;

  Handle<reco::TrackCollection> trackOrigH;
  iEvent.getByToken(inputTrackOrig_, trackOrigH);


  // loop over gen particles

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);
    
  edm::ESHandle<TrackerTopology> trackerTopology;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopology);
  
  
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrh;
  iSetup.get<TransientRecHitRecord>().get("WithAngleAndTemplate",ttrh);
  
  ESHandle<Propagator> thePropagator;
  iSetup.get<TrackingComponentsRecord>().get("RungeKuttaTrackerPropagator", thePropagator);
//   iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", thePropagator);
//   iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterialParabolicMf", thePropagator);
//   iSetup.get<TrackingComponentsRecord>().get("Geant4ePropagator", thePropagator);
  const MagneticField* field = thePropagator->magneticField();
  
  ESHandle<Propagator> theAnalyticPropagator;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", theAnalyticPropagator);

  
  Handle<std::vector<reco::GenParticle>> genPartCollection;
  if (doGen_) {
    iEvent.getByToken(GenParticlesToken_, genPartCollection);
  }
  
  std::unique_ptr<PropagatorWithMaterial> fPropagator(static_cast<PropagatorWithMaterial*>(thePropagator->clone()));
  fPropagator->setPropagationDirection(alongMomentum);
  
  std::unique_ptr<PropagatorWithMaterial> fAnalyticPropagator(static_cast<PropagatorWithMaterial*>(theAnalyticPropagator->clone()));
  fAnalyticPropagator->setPropagationDirection(alongMomentum);
  
  KFUpdator updator;
  TkClonerImpl const& cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(ttrh.product())->cloner();
  
  edm::ESHandle<TransientTrackBuilder> TTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", TTBuilder);
  KinematicParticleFactoryFromTransientTrack pFactory;
  
  constexpr double mmu = 0.1056583745;
  constexpr double mmuerr = 0.0000000024;
  
  
  VectorXd gradfull;
  MatrixXd hessfull;
  
  VectorXd dxfull;
  MatrixXd dxdparms;
  VectorXd grad;
  MatrixXd hess;
  LDLT<MatrixXd> Cinvd;
  
  std::array<MatrixXd, 2> jacarr;
  
  // loop over combinatorics of track pairs
  for (auto itrack = trackOrigH->begin(); itrack != trackOrigH->end(); ++itrack) {
    const reco::TransientTrack itt = TTBuilder->build(*itrack);
    for (auto jtrack = itrack + 1; jtrack != trackOrigH->end(); ++jtrack) {
      const reco::TransientTrack jtt = TTBuilder->build(*jtrack);
      
      // common vertex fit
      std::vector<RefCountedKinematicParticle> parts;
      
      float masserr = mmuerr;
      float chisq = 0.;
      float ndf = 0.;
      parts.push_back(pFactory.particle(itt, mmu, chisq, ndf, masserr));
      parts.push_back(pFactory.particle(jtt, mmu, chisq, ndf, masserr));
      
      KinematicParticleVertexFitter vtxFitter;
      RefCountedKinematicTree kinTree = vtxFitter.fit(parts);
      
      if (kinTree->isEmpty() || !kinTree->isConsistent()) {
        continue;
      }
      
      if (false) {
        // debug output
        kinTree->movePointerToTheTop();
        
        RefCountedKinematicParticle dimu_kinfit = kinTree->currentParticle();
        RefCountedKinematicVertex dimu_vertex = kinTree->currentDecayVertex();
        
        std::cout << dimu_kinfit->currentState().mass() << std::endl;
        std::cout << dimu_vertex->position() << std::endl;
      }
      
      std::vector<RefCountedKinematicParticle> outparts = kinTree->finalStateParticles();
      
      std::array<TransientTrackingRecHit::RecHitContainer, 2> hitsarr;
      
      // prepare hits
      for (unsigned int id = 0; id < 2; ++id) {
        const reco::Track &track = id == 0 ? *itrack : *jtrack;
        auto &hits = hitsarr[id];
        hits.reserve(track.recHitsSize());
      
      
        for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
          const GeomDet* detectorG = globalGeometry->idToDet((*it)->geographicalId());
          const GluedGeomDet* detglued = dynamic_cast<const GluedGeomDet*>(detectorG);
          
          // split matched invalid hits
          if (detglued != nullptr && !(*it)->isValid()) {
            bool order = detglued->stereoDet()->surface().position().mag() > detglued->monoDet()->surface().position().mag();
            const GeomDetUnit* detinner = order ? detglued->monoDet() : detglued->stereoDet();
            const GeomDetUnit* detouter = order ? detglued->stereoDet() : detglued->monoDet();
            
            hits.push_back(TrackingRecHit::RecHitPointer(new InvalidTrackingRecHit(*detinner, (*it)->type())));
            hits.push_back(TrackingRecHit::RecHitPointer(new InvalidTrackingRecHit(*detouter, (*it)->type())));
          }
          else {
            // apply hit quality criteria
            const bool ispixel = GeomDetEnumerators::isTrackerPixel(detectorG->subDetector());
            bool hitquality = false;
            if (applyHitQuality_ && (*it)->isValid()) {
              const TrackerSingleRecHit* tkhit = dynamic_cast<const TrackerSingleRecHit*>(*it);
              assert(tkhit != nullptr);
              
              if (ispixel) {
                const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>(tkhit);
                const SiPixelCluster& cluster = *tkhit->cluster_pixel();
                assert(pixhit != nullptr);
                
                hitquality = !pixhit->isOnEdge() && cluster.sizeX() > 1;
              }
              else {
                hitquality = true;
              }
              
            }
            else {
              hitquality = true;
            }

            
            if (hitquality) {
              hits.push_back((*it)->cloneForFit(*detectorG));              
            }
            else {
              hits.push_back(TrackingRecHit::RecHitPointer(new InvalidTrackingRecHit(*detectorG, TrackingRecHit::inactive)));
            }
          }          
        }
      }
      
      unsigned int nhits = 0;
      unsigned int nvalid = 0;
      unsigned int nvalidpixel = 0;
      unsigned int nvalidalign2d = 0;
      // second loop to count hits
      for (auto const &hits : hitsarr) {
        for (auto const &hit : hits) {
          ++nhits;
          if (hit->isValid()) {
            ++nvalid;
            
            const bool ispixel = GeomDetEnumerators::isTrackerPixel(hit->det()->subDetector());
            if (ispixel) {
              ++nvalidpixel;
            }
            
            
            const bool align2d = detidparms.count(std::make_pair(1, hit->geographicalId()));
            if (align2d) {
              ++nvalidalign2d;
            }
          } 
        }
      }
      

      
      const unsigned int nparsAlignment = 2*nvalid + nvalidalign2d;
      const unsigned int nparsBfield = nhits;
      const unsigned int nparsEloss = nhits - 2;
      const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
      
      const unsigned int nstateparms = 5 + 3*nhits - 2;
//       const unsigned int nstateparms = 3 + 2*nhits + 2 + nhits - 2;
//       const unsigned int nstateparms = 2 + 2*nhits + nhits - 1 + 1;
//       const unsigned int nstateparms = 3*(nhits+1) - 2 - 1;
//       const unsigned int nstateparms = 3 + 2*nhits + nhits + 2 - 2
      const unsigned int nparmsfull = nstateparms + npars;
      
      gradfull = VectorXd::Zero(nparmsfull);
      hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);

      globalidxv.clear();
      globalidxv.resize(npars, 0);
      
      nParms = npars;
      if (fillTrackTree_) {
        tree->SetBranchAddress("globalidxv", globalidxv.data());
      }
      
      bool valid = true;
      
      unsigned int trackstateidx = 3;
      unsigned int parmidx = 0;
      unsigned int alignmentparmidx = 0;
      
      for (unsigned int id = 0; id < 2; ++id) {
        FreeTrajectoryState refFts = outparts[id]->currentState().freeTrajectoryState();
        auto &hits = hitsarr[id];
        
        const unsigned int tracknhits = hits.size();
        
        auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, *hits[0]->surface());
  //       auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, *beampipe);
        if (!propresult.first.isValid()) {
          std::cout << "Abort: Propagation of reference state Failed!" << std::endl;
          valid = false;
          break;
        }
        
  //       std::cout << "position on beampipe " << propresult.first.globalParameters().position() << std::endl;
        
        // cartesian to curvilinear jacobian, needed for cartesian parameterization of common vertex position
        JacobianCartesianToCurvilinear cart2curvref(refFts.parameters());
        auto const &jacCart2Curvref = Map<const Matrix<double, 5, 6, RowMajor>>(cart2curvref.jacobian().Array());
        
        const Matrix<double, 5, 6> FdFp = curv2curvTransportJacobian(refFts, propresult, false);

        const Matrix<double, 2, 2> J = FdFp.block<2, 2>(3, 3);
        // (du/dalphap)^-1
        const Matrix<double, 2, 2> Sinv = FdFp.block<2, 2>(3, 1).inverse();
        // du/dqopp
        const Matrix<double, 2, 1> D = FdFp.block<2, 1>(3, 0);
        // du/dBp
//         Matrix<double, 2, 1> Bpref = FdFp.block<2, 1>(3, 5);
        // d(dxy, dsz) / dvtx
        const Matrix<double, 2, 3> Vref = jacCart2Curvref.bottomLeftCorner<2,3>();

        
        MatrixXd &jac = jacarr[id];
        jac = MatrixXd::Zero(5, nstateparms);
        
        // dqop / dqop_0
        jac(0, trackstateidx) = 1.;
        // d(lambda, phi) / dqop_0
        jac.block<2, 1>(1, trackstateidx) = -Sinv*D;
        // d(lambda, phi) / dvtx
        jac.block<2, 3>(1, 0) = -Sinv*J*Vref;
        // d(lambda, phi)_i/ d(dxy, dsz)_1
        jac.block<2, 2>(1, trackstateidx + 1) = Sinv;
        // d(dxy, dsz)/dvtx
        jac.block<2, 3>(3, 0) = Vref;
        
        
        //TODO add state jacobian stuff here
          
        Matrix<double, 5, 6> FdFm = curv2curvTransportJacobian(refFts, propresult, true);
        
        for (unsigned int ihit = 0; ihit < hits.size(); ++ihit) {
  //         std::cout << "ihit " << ihit << std::endl;
          auto const& hit = hits[ihit];
                  
          TrajectoryStateOnSurface updtsos = propresult.first;
          
          //apply measurement update if applicable
  //         std::cout << "constructing preciseHit" << std::endl;
          auto const& preciseHit = hit->isValid() ? cloner.makeShared(hit, updtsos) : hit;
          if (hit->isValid() && !preciseHit->isValid()) {
            std::cout << "Abort: Failed updating hit" << std::endl;
            valid = false;
            break;
          }
          
  //         const uint32_t gluedid = trackerTopology->glued(preciseHit->det()->geographicalId());
  //         const bool isglued = gluedid != 0;
  //         const DetId parmdetid = isglued ? DetId(gluedid) : preciseHit->geographicalId();
  //         const bool align2d = detidparms.count(std::make_pair(1, parmdetid));
  //         const GeomDet* parmDet = isglued ? globalGeometry->idToDet(parmdetid) : preciseHit->det();
          
          const bool align2d = detidparms.count(std::make_pair(1, preciseHit->geographicalId()));

          
          // compute convolution correction in local coordinates (BEFORE material effects are applied)
  //         const Matrix<double, 2, 1> dxlocalconv = localPositionConvolution(updtsos);
          
          // curvilinear to local jacobian
          JacobianCurvilinearToLocal curv2localm(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
          const AlgebraicMatrix55& curv2localjacm = curv2localm.jacobian();
          const Matrix<double, 5, 5> Hm = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacm.Array()); 
          
          //energy loss jacobian
          const Matrix<double, 5, 6> EdE = materialEffectsJacobian(updtsos, fPropagator->materialEffectsUpdator());
        
          //process noise jacobians
//           const std::array<Matrix<double, 5, 5>, 5> dQs = processNoiseJacobians(updtsos, fPropagator->materialEffectsUpdator());
          
          //TODO update code to allow doing this in one step with nominal update
          //temporary tsos to extract process noise without loss of precision
          TrajectoryStateOnSurface tmptsos(updtsos);
          tmptsos.update(tmptsos.localParameters(),
                          LocalTrajectoryError(0.,0.,0.,0.,0.),
                          tmptsos.surface(),
                          tmptsos.magneticField(),
                          tmptsos.surfaceSide());
          
          //apply the state update from the material effects
          bool ok = fPropagator->materialEffectsUpdator().updateStateInPlace(tmptsos, alongMomentum);
          if (!ok) {
            std::cout << "Abort: material update failed" << std::endl;
            valid = false;
            break;
          }
          
          const AlgebraicVector5 dxeloss = tmptsos.localParameters().vector() - updtsos.localParameters().vector();
          
          // compute convolution effects
  //         const AlgebraicVector5 dlocalconv = localMSConvolution(updtsos, fPropagator->materialEffectsUpdator());
          
  //         const GlobalPoint updtsospos = updtsos.globalParameters().position();
  //         std::cout << "before material update: " << updtsos.globalParameters().position() << " " << updtsos.globalParameters().momentum() << std::endl;
          ok = fPropagator->materialEffectsUpdator().updateStateInPlace(updtsos, alongMomentum);
          if (!ok) {
            std::cout << "Abort: material update failed" << std::endl;
            valid = false;
            break;
          }
  //         std::cout << "after material update: " << updtsos.globalParameters().position() << " " << updtsos.globalParameters().momentum() << std::endl;
          
          
  //         std::cout << "local parameters" << std::endl;
  //         std::cout << updtsos.localParameters().vector() << std::endl;
  //         std::cout << "dlocalconv" << std::endl;
  //         std::cout << dlocalconv << std::endl;
  //         
  //         // apply convolution effects
  //         const LocalTrajectoryParameters localupd(updtsos.localParameters().vector() + dlocalconv,
  //                                                  updtsos.localParameters().pzSign());
  //         updtsos.update(localupd,
  // //                        updtsos.localError(),
  //                        updtsos.surface(),
  //                        updtsos.magneticField(),
  //                        updtsos.surfaceSide());
          
          //get the process noise matrix
          AlgebraicMatrix55 const Qmat = tmptsos.localError().matrix();
          const Map<const Matrix<double, 5, 5, RowMajor>>Q(Qmat.Array());
  //         std::cout<< "Q" << std::endl;
  //         std::cout<< Q << std::endl;
          

          // FIXME this is not valid for multiple iterations
          // curvilinear to local jacobian
          JacobianCurvilinearToLocal curv2localp(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
          const AlgebraicMatrix55& curv2localjacp = curv2localp.jacobian();
          const Matrix<double, 5, 5> Hp = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacp.Array());
          
          if (ihit < (tracknhits-1)) {
            AlgebraicVector5 idx0(0., 0., 0., 0., 0.);
            const Vector5d dx0 = Map<const Vector5d>(idx0.Array());
            
            
            propresult = fPropagator->geometricalPropagator().propagateWithPath(updtsos, *hits[ihit+1]->surface());
            if (!propresult.first.isValid()) {
              std::cout << "Abort: Propagation Failed!" << std::endl;
              valid = false;
              break;
            }
            
            //forward propagation jacobian
            const Matrix<double, 5, 6> FdFp = curv2curvTransportJacobian(*updtsos.freeState(), propresult, false);

            //TODO restore unused stuff below? (related to statejac stuff?)
//             Matrix<double, 2, 2> J = FdFp.block<2, 2>(3, 3);
//             // (du/dalphap)^-1
//             Matrix<double, 2, 2> Sinv = FdFp.block<2, 2>(3, 1).inverse();
//             // du/dqopp
//             Matrix<double, 2, 1> D = FdFp.block<2, 1>(3, 0);
            
            if (ihit == 0) {
              constexpr unsigned int nvtxstate = 3;
              constexpr unsigned int nlocalstate = 6;
              constexpr unsigned int nlocalbfield = 2;
              constexpr unsigned int nlocaleloss = 1;
              constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
              
              constexpr unsigned int nlocal = nvtxstate + nlocalstate + nlocalparms;
              
              constexpr unsigned int localvtxidx = 0;
              constexpr unsigned int localstateidx = localvtxidx + nvtxstate;
              constexpr unsigned int localparmidx = localstateidx + nlocalstate;
              
              constexpr unsigned int fullvtxidx = 0;
              const unsigned int fullstateidx = trackstateidx;
              const unsigned int fullparmidx = nstateparms + parmidx;
              
              using MSScalar = AANT<double, nlocal>;
              
              // individual pieces, now starting to cast to active scalars for autograd,
              
              // dum/dvtx
              const Matrix<MSScalar, 2, 3> Vm = jacCart2Curvref.bottomLeftCorner<2,3>().cast<MSScalar>();
              
              // as in eq (3) of https://doi.org/10.1016/j.cpc.2011.03.017
              
              // du/dum
              const Matrix<MSScalar, 2, 2> Jm = FdFm.block<2, 2>(3, 3).cast<MSScalar>();
              // (du/dalpham)^-1
              const Matrix<MSScalar, 2, 2> Sinvm = FdFm.block<2, 2>(3, 1).inverse().cast<MSScalar>();
              // du/dqopm
              const Matrix<MSScalar, 2, 1> Dm = FdFm.block<2, 1>(3, 0).cast<MSScalar>();
              // du/dBm
              const Matrix<MSScalar, 2, 1> Bm = FdFm.block<2, 1>(3, 5).cast<MSScalar>();

              // du/dup
              const Matrix<MSScalar, 2, 2> Jp = FdFp.block<2, 2>(3, 3).cast<MSScalar>();
              // (du/dalphap)^-1
              const Matrix<MSScalar, 2, 2> Sinvp = FdFp.block<2, 2>(3, 1).inverse().cast<MSScalar>();
              // du/dqopp
              const Matrix<MSScalar, 2, 1> Dp = FdFp.block<2, 1>(3, 0).cast<MSScalar>();
              // du/dBp
              const Matrix<MSScalar, 2, 1> Bp = FdFp.block<2, 1>(3, 5).cast<MSScalar>();
              
              const MSScalar Eqop(EdE(0,0));
              const Matrix<MSScalar, 1, 2> Ealpha = EdE.block<1, 2>(0, 1).cast<MSScalar>();
              const MSScalar dE(EdE(0,5));
              
              const MSScalar muE(dxeloss[0]);
              
              //energy loss inverse variance
              const MSScalar invSigmaE(1./Q(0,0));
              
              // multiple scattering inverse covariance
              const Matrix<MSScalar, 2, 2> Qinvms = Q.block<2,2>(1,1).inverse().cast<MSScalar>();
              
              
              // initialize active scalars for common vertex parameters
              Matrix<MSScalar, 3, 1> dvtx = Matrix<MSScalar, 3, 1>::Zero();
              for (unsigned int j=0; j<dvtx.size(); ++j) {
                init_twice_active_var(dvtx[j], nlocal, localvtxidx + j);
              }

              // initialize active scalars for state parameters
              MSScalar dqopm(0.);
              init_twice_active_var(dqopm, nlocal, localstateidx);
              
              Matrix<MSScalar, 2, 1> du = Matrix<MSScalar, 2, 1>::Zero();
              for (unsigned int j=0; j<du.size(); ++j) {
                init_twice_active_var(du[j], nlocal, localstateidx + 1 + j);
              }
              
              MSScalar dqop(0.);
              init_twice_active_var(dqop, nlocal, localstateidx + 3);

              Matrix<MSScalar, 2, 1> dup = Matrix<MSScalar, 2, 1>::Zero();
              for (unsigned int j=0; j<dup.size(); ++j) {
                init_twice_active_var(dup[j], nlocal, localstateidx + 4 + j);
              }
              
              // initialize active scalars for correction parameters

              MSScalar dbeta(0.);
              init_twice_active_var(dbeta, nlocal, localparmidx);
              
              MSScalar dxi(0.);
              init_twice_active_var(dxi, nlocal, localparmidx + 1);
              
              MSScalar dbetap(0.);
              init_twice_active_var(dbetap, nlocal, localparmidx + 2);
              
              //multiple scattering kink term
              
              const Matrix<MSScalar, 2, 1> dum = Vm*dvtx;
              
              const Matrix<MSScalar, 2, 2> Halphalamphim = Hm.block<2,2>(1, 1).cast<MSScalar>();
              const Matrix<MSScalar, 2, 2> Halphaum = Hm.block<2,2>(1, 3).cast<MSScalar>();
              
              const Matrix<MSScalar, 2, 2> Halphalamphip = Hp.block<2,2>(1, 1).cast<MSScalar>();
              const Matrix<MSScalar, 2, 2> Halphaup = Hp.block<2,2>(1, 3).cast<MSScalar>();
              
              const Matrix<MSScalar, 2, 1> dalpha0 = dx0.segment<2>(1).cast<MSScalar>();
    
              const Matrix<MSScalar, 2, 1> dlamphim = Sinvm*(dum - Jm*du - Dm*dqopm - Bm*dbeta);
              const Matrix<MSScalar, 2, 1> dlamphip = Sinvp*(dup - Jp*du - Dp*dqop - Bp*dbetap);
              
              const Matrix<MSScalar, 2, 1> dalpham = Halphalamphim*dlamphim + Halphaum*du;
              const Matrix<MSScalar, 2, 1> dalphap = Halphalamphip*dlamphip + Halphaup*du;
              
              const MSScalar deloss0(dx0[0]);
              
              const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
              const MSScalar chisqms = dms.transpose()*Qinvms*dms;
              //energy loss term
              
              
              const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
              const MSScalar chisqeloss = deloss*invSigmaE*deloss;
              
              const MSScalar chisq = chisqms + chisqeloss;
              
              auto const& gradlocal = chisq.value().derivatives();
              //fill local hessian
              Matrix<double, nlocal, nlocal> hesslocal;
              for (unsigned int j=0; j<nlocal; ++j) {
                hesslocal.row(j) = chisq.derivatives()[j].derivatives();
              }
              
//               std::cout << "first hit gradlocal" << std::endl;
//               std::cout << gradlocal << std::endl;
//               
//               std::cout << "first hit hesslocal" << std::endl;
//               std::cout << hesslocal << std::endl;
              
              //Fill global grad and hess (upper triangular blocks only)
              gradfull.segment<nvtxstate>(fullvtxidx) += gradlocal.segment<nvtxstate>(localvtxidx);
              gradfull.segment<nlocalstate>(fullstateidx) += gradlocal.segment<nlocalstate>(localstateidx);
              gradfull.segment<nlocalparms>(fullparmidx) += gradlocal.segment<nlocalparms>(localparmidx);
              
              hessfull.block<nvtxstate, nvtxstate>(fullvtxidx, fullvtxidx) += hesslocal.block<nvtxstate, nvtxstate>(localvtxidx, localvtxidx);
              hessfull.block<nvtxstate, nlocalstate>(fullvtxidx, fullstateidx) += hesslocal.block<nvtxstate, nlocalstate>(localvtxidx, localstateidx);
              hessfull.block<nvtxstate, nlocalparms>(fullvtxidx, fullparmidx) += hesslocal.block<nvtxstate, nlocalparms>(localvtxidx, localparmidx);
              hessfull.block<nlocalstate, nlocalstate>(fullstateidx, fullstateidx) += hesslocal.block<nlocalstate,nlocalstate>(localstateidx, localstateidx);
              hessfull.block<nlocalstate, nlocalparms>(fullstateidx, fullparmidx) += hesslocal.block<nlocalstate, nlocalparms>(localstateidx, localparmidx);
              hessfull.block<nlocalparms, nlocalparms>(fullparmidx, fullparmidx) += hesslocal.block<nlocalparms, nlocalparms>(localparmidx, localparmidx);
              
//               std::cout << "first hit, parm idx = " << parmidx << std::endl;
              
              const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(6,preciseHit->geographicalId()));
              globalidxv[parmidx] = bfieldglobalidx;
              parmidx++;
              
              const unsigned int elossglobalidx = detidparms.at(std::make_pair(7,preciseHit->geographicalId()));
              globalidxv[parmidx] = elossglobalidx;
              parmidx++;
              
            }
            else {
              //TODO statejac stuff
              
              constexpr unsigned int nlocalstate = 8;
              constexpr unsigned int nlocalbfield = 2;
              constexpr unsigned int nlocaleloss = 1;
              constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
              
              constexpr unsigned int nlocal = nlocalstate + nlocalparms;
              
              constexpr unsigned int localstateidx = 0;
              constexpr unsigned int localparmidx = localstateidx + nlocalstate;
              
              const unsigned int fullstateidx = trackstateidx + 1 + 3*(ihit - 1);
              const unsigned int fullparmidx = nstateparms + parmidx;
              
              using MSScalar = AANT<double, nlocal>;

              
              // individual pieces, now starting to cast to active scalars for autograd,
              // as in eq (3) of https://doi.org/10.1016/j.cpc.2011.03.017
              // du/dum
              const Matrix<MSScalar, 2, 2> Jm = FdFm.block<2, 2>(3, 3).cast<MSScalar>();
              // (du/dalpham)^-1
              const Matrix<MSScalar, 2, 2> Sinvm = FdFm.block<2, 2>(3, 1).inverse().cast<MSScalar>();
              // du/dqopm
              const Matrix<MSScalar, 2, 1> Dm = FdFm.block<2, 1>(3, 0).cast<MSScalar>();
              // du/dBm
              const Matrix<MSScalar, 2, 1> Bm = FdFm.block<2, 1>(3, 5).cast<MSScalar>();

              // du/dup
              const Matrix<MSScalar, 2, 2> Jp = FdFp.block<2, 2>(3, 3).cast<MSScalar>();
              // (du/dalphap)^-1
              const Matrix<MSScalar, 2, 2> Sinvp = FdFp.block<2, 2>(3, 1).inverse().cast<MSScalar>();
              // du/dqopp
              const Matrix<MSScalar, 2, 1> Dp = FdFp.block<2, 1>(3, 0).cast<MSScalar>();
              // du/dBp
              const Matrix<MSScalar, 2, 1> Bp = FdFp.block<2, 1>(3, 5).cast<MSScalar>();
              
              const MSScalar Eqop(EdE(0,0));
              const Matrix<MSScalar, 1, 2> Ealpha = EdE.block<1, 2>(0, 1).cast<MSScalar>();
              const MSScalar dE(EdE(0,5));
              
              const MSScalar muE(dxeloss[0]);
              
              //energy loss inverse variance
              const MSScalar invSigmaE(1./Q(0,0));
              
              // multiple scattering inverse covariance
              const Matrix<MSScalar, 2, 2> Qinvms = Q.block<2,2>(1,1).inverse().cast<MSScalar>();
              
              // initialize active scalars for state parameters
              Matrix<MSScalar, 2, 1> dum = Matrix<MSScalar, 2, 1>::Zero();
              //suppress gradients of reference point parameters when fitting with gen constraint
              for (unsigned int j=0; j<dum.size(); ++j) {
                init_twice_active_var(dum[j], nlocal, localstateidx + j);
              }
              
              MSScalar dqopm(0.);
              init_twice_active_var(dqopm, nlocal, localstateidx + 2);
              
              Matrix<MSScalar, 2, 1> du = Matrix<MSScalar, 2, 1>::Zero();
              for (unsigned int j=0; j<du.size(); ++j) {
                init_twice_active_var(du[j], nlocal, localstateidx + 3 + j);
              }
              
              MSScalar dqop(0.);
              init_twice_active_var(dqop, nlocal, localstateidx + 5);

              Matrix<MSScalar, 2, 1> dup = Matrix<MSScalar, 2, 1>::Zero();
              for (unsigned int j=0; j<dup.size(); ++j) {
                init_twice_active_var(dup[j], nlocal, localstateidx + 6 + j);
              }
              
              // initialize active scalars for correction parameters

              MSScalar dbeta(0.);
              init_twice_active_var(dbeta, nlocal, localparmidx);
              
              MSScalar dxi(0.);
              init_twice_active_var(dxi, nlocal, localparmidx + 1);
              
              MSScalar dbetap(0.);
              init_twice_active_var(dbetap, nlocal, localparmidx + 2);
              
              //multiple scattering kink term
              
              Matrix<MSScalar, 2, 2> Halphalamphim = Hm.block<2,2>(1, 1).cast<MSScalar>();
              Matrix<MSScalar, 2, 2> Halphaum = Hm.block<2,2>(1, 3).cast<MSScalar>();
              
              Matrix<MSScalar, 2, 2> Halphalamphip = Hp.block<2,2>(1, 1).cast<MSScalar>();
              Matrix<MSScalar, 2, 2> Halphaup = Hp.block<2,2>(1, 3).cast<MSScalar>();
              
              const Matrix<MSScalar, 2, 1> dalpha0 = dx0.segment<2>(1).cast<MSScalar>();
    
              const Matrix<MSScalar, 2, 1> dlamphim = Sinvm*(dum - Jm*du - Dm*dqopm - Bm*dbeta);
              const Matrix<MSScalar, 2, 1> dlamphip = Sinvp*(dup - Jp*du - Dp*dqop - Bp*dbetap);
              
              const Matrix<MSScalar, 2, 1> dalpham = Halphalamphim*dlamphim + Halphaum*du;
              const Matrix<MSScalar, 2, 1> dalphap = Halphalamphip*dlamphip + Halphaup*du;
              
              const MSScalar deloss0(dx0[0]);
              
              const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
              const MSScalar chisqms = dms.transpose()*Qinvms*dms;
              //energy loss term
              
              
              const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
              const MSScalar chisqeloss = deloss*invSigmaE*deloss;
              
              const MSScalar chisq = chisqms + chisqeloss;
                
              auto const& gradlocal = chisq.value().derivatives();
              //fill local hessian
              Matrix<double, nlocal, nlocal> hesslocal;
              for (unsigned int j=0; j<nlocal; ++j) {
                hesslocal.row(j) = chisq.derivatives()[j].derivatives();
              }
              
              //fill global gradient
              gradfull.segment<nlocalstate>(fullstateidx) += gradlocal.head<nlocalstate>();
              gradfull.segment<nlocalparms>(fullparmidx) += gradlocal.segment<nlocalparms>(localparmidx);

              //fill global hessian (upper triangular blocks only)
              hessfull.block<nlocalstate,nlocalstate>(fullstateidx, fullstateidx) += hesslocal.topLeftCorner<nlocalstate,nlocalstate>();
              hessfull.block<nlocalstate,nlocalparms>(fullstateidx, fullparmidx) += hesslocal.topRightCorner<nlocalstate, nlocalparms>();
              hessfull.block<nlocalparms, nlocalparms>(fullparmidx, fullparmidx) += hesslocal.bottomRightCorner<nlocalparms, nlocalparms>();
              
//               std::cout << "intermediate hit, parm idx = " << parmidx << std::endl;
              
              const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(6,preciseHit->geographicalId()));
              globalidxv[parmidx] = bfieldglobalidx;
              parmidx++;
              
              const unsigned int elossglobalidx = detidparms.at(std::make_pair(7,preciseHit->geographicalId()));
              globalidxv[parmidx] = elossglobalidx;
              parmidx++;
              
              
            }
            
            //backwards propagation jacobian (local to local) to be used at the next layer
            FdFm = curv2curvTransportJacobian(*updtsos.freeState(), propresult, true);
          }
          else {
//             std::cout << "last hit, parm idx = " << parmidx << std::endl;
            
            const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(6,preciseHit->geographicalId()));
            globalidxv[parmidx] = bfieldglobalidx;
            parmidx++; 
          }
          
          if (preciseHit->isValid()) {

            auto fillAlignGrads = [&](auto Nalign) {
              constexpr unsigned int nlocalstate = 2;
              constexpr unsigned int localstateidx = 0;
              constexpr unsigned int localalignmentidx = nlocalstate;
              constexpr unsigned int localparmidx = localalignmentidx;

              // abusing implicit template argument to pass
              // a template value via std::integral_constant
              constexpr unsigned int nlocalalignment = Nalign();
              constexpr unsigned int nlocalparms = nlocalalignment;
              constexpr unsigned int nlocal = nlocalstate + nlocalparms;
              
//               std::cout << "idx = " << id << ", ihit = " << ihit << ", alignmentparmidx = " << alignmentparmidx << ", nlocalalignment = " << nlocalalignment << std::endl;

              using AlignScalar = AANT<double, nlocal>;
              
              const unsigned int fullstateidx = trackstateidx + 1 + 3*ihit;
              const unsigned int fullparmidx = nstateparms + nparsBfield + nparsEloss + alignmentparmidx;

              const bool ispixel = GeomDetEnumerators::isTrackerPixel(preciseHit->det()->subDetector());

              //TODO add hit validation stuff
              //TODO add simhit stuff
              
              Matrix<AlignScalar, 2, 2> Hu = Hp.bottomRightCorner<2,2>().cast<AlignScalar>();

              Matrix<AlignScalar, 2, 1> dy0;
              Matrix<AlignScalar, 2, 2> Vinv;
              // rotation from module to strip coordinates
  //             Matrix<AlignScalar, 2, 2> R;
              Matrix2d R;
              if (preciseHit->dimension() == 1) {
                dy0[0] = AlignScalar(preciseHit->localPosition().x() - updtsos.localPosition().x());
                dy0[1] = AlignScalar(0.);
                
  //               bool simvalid = false;
  //               for (auto const& simhith : simHits) {
  //                 for (const PSimHit& simHit : *simhith) {
  //                   if (simHit.detUnitId() == preciseHit->geographicalId()) {                      
  //                     
  //                     dy0[0] = AlignScalar(simHit.localPosition().x() - updtsos.localPosition().x());
  //                     
  //                     simvalid = true;
  //                     break;
  //                   }
  //                 }
  //                 if (simvalid) {
  //                   break;
  //                 }
  //               }
                
                Vinv = Matrix<AlignScalar, 2, 2>::Zero();
                Vinv(0,0) = 1./preciseHit->localPositionError().xx();
                
  //               R = Matrix<AlignScalar, 2, 2>::Identity();
                R = Matrix2d::Identity();
              }
              else {
                // 2d hit
                Matrix2d iV;
                iV << preciseHit->localPositionError().xx(), preciseHit->localPositionError().xy(),
                      preciseHit->localPositionError().xy(), preciseHit->localPositionError().yy();
                if (ispixel) {
                  //take 2d hit as-is for pixels
                  dy0[0] = AlignScalar(preciseHit->localPosition().x() - updtsos.localPosition().x());
                  dy0[1] = AlignScalar(preciseHit->localPosition().y() - updtsos.localPosition().y());
                
                  Vinv = iV.inverse().cast<AlignScalar>();
                  //FIXME various temporary hacks;
                  
  //                 dy0[1] = AlignScalar(0.);
  //                 Vinv = Matrix<AlignScalar, 2, 2>::Zero();
  //                 Vinv(0,0) = 1./preciseHit->localPositionError().xx();
                  
  //                 if (GeomDetEnumerators::isEndcap(preciseHit->det()->subDetector())) {
  //                 if (GeomDetEnumerators::isBarrel(preciseHit->det()->subDetector())) {
  //                   PXBDetId detidtest(preciseHit->det()->geographicalId());
  //                   int layertest = detidtest.layer();
  //                   
  //                   if (layertest > 1) {
  //                     Vinv = Matrix<AlignScalar, 2, 2>::Zero();
  //                   }
  //                   
  // //                   Vinv = Matrix<AlignScalar, 2, 2>::Zero();
  // //                   dy0[0] = AlignScalar(0.);
  // //                   dy0[1] = AlignScalar(0.);
  //                 }
                  
  //                 bool simvalid = false;
  //                 for (auto const& simhith : simHits) {
  //                   for (const PSimHit& simHit : *simhith) {
  //                     if (simHit.detUnitId() == preciseHit->geographicalId()) {                      
  //                       
  //                       if (GeomDetEnumerators::isBarrel(preciseHit->det()->subDetector())) {
  //                         dy0[0] = AlignScalar(simHit.localPosition().x() - updtsos.localPosition().x());
  //                         dy0[1] = AlignScalar(simHit.localPosition().y() - updtsos.localPosition().y());
  //                       }
  //                       
  // //                       dy0[1] = AlignScalar(0.);
  //                       
  //                       simvalid = true;
  //                       break;
  //                     }
  //                   }
  //                   if (simvalid) {
  //                     break;
  //                   }
  //                 }
                  
                  
  //                 R = Matrix<AlignScalar, 2, 2>::Identity();
                  R = Matrix2d::Identity();
                }
                else {
                  // diagonalize and take only smallest eigenvalue for 2d hits in strip wedge modules,
                  // since the constraint parallel to the strip is spurious
                  SelfAdjointEigenSolver<Matrix2d> eigensolver(iV);
  //                 const Matrix2d& v = eigensolver.eigenvectors();
                  R = eigensolver.eigenvectors().transpose();
                  if (R(0,0) < 0.) {
                    R.row(0) *= -1.;
                  }
                  if (R(1,1) <0.) {
                    R.row(1) *= -1.;
                  }
                  
                  Matrix<double, 2, 1> dy0local;
                  dy0local[0] = preciseHit->localPosition().x() - updtsos.localPosition().x();
                  dy0local[1] = preciseHit->localPosition().y() - updtsos.localPosition().y();
                  
  //                 bool simvalid = false;
  //                 for (auto const& simhith : simHits) {
  //                   for (const PSimHit& simHit : *simhith) {
  //                     if (simHit.detUnitId() == preciseHit->geographicalId()) {                      
  //                       
  //                       dy0local[0] = simHit.localPosition().x() - updtsos.localPosition().x();
  //                       dy0local[1] = simHit.localPosition().y() - updtsos.localPosition().y();
  //                       
  //                       simvalid = true;
  //                       break;
  //                     }
  //                   }
  //                   if (simvalid) {
  //                     break;
  //                   }
  //                 }
                  
                  const Matrix<double, 2, 1> dy0eig = R*dy0local;
                  
                  //TODO deal properly with rotations (rotate back to module local coords?)
                  dy0[0] = AlignScalar(dy0eig[0]);
                  dy0[1] = AlignScalar(0.);
                  
                  Vinv = Matrix<AlignScalar, 2, 2>::Zero();
                  Vinv(0,0) = AlignScalar(1./eigensolver.eigenvalues()[0]);      
                  
  //                 R = v.transpose().cast<AlignScalar>();
                  
                }
              }
              
//               rxfull.row(ivalidhit) = R.row(0).cast<float>();
//               ryfull.row(ivalidhit) = R.row(1).cast<float>();
              
//               validdxeigjac.block<2,2>(2*ivalidhit, 3*(ihit+1)) = R*Hp.bottomRightCorner<2,2>();
              
              const Matrix<AlignScalar, 2, 2> Ralign = R.cast<AlignScalar>();
              
              Matrix<AlignScalar, 2, 1> dx = Matrix<AlignScalar, 2, 1>::Zero();
              for (unsigned int j=0; j<dx.size(); ++j) {
                init_twice_active_var(dx[j], nlocal, localstateidx + j);
              }

              Matrix<AlignScalar, 6, 1> dalpha = Matrix<AlignScalar, 6, 1>::Zero();
              // order in which to use parameters, especially relevant in case nlocalalignment < 6
              constexpr std::array<unsigned int, 6> alphaidxs = {{5, 0, 1, 2, 3, 4}};
              for (unsigned int idim=0; idim<nlocalalignment; ++idim) {
  //               init_twice_active_var(dalpha[idim], nlocal, localalignmentidx+idim);
                init_twice_active_var(dalpha[alphaidxs[idim]], nlocal, localalignmentidx+idim);
              }
              
              // alignment jacobian
              Matrix<AlignScalar, 2, 6> A = Matrix<AlignScalar, 2, 6>::Zero();

                          
              // dx/dx
              A(0,0) = AlignScalar(1.);
              // dy/dy
              A(1,1) = AlignScalar(1.);
              // dx/dz
              A(0,2) = updtsos.localParameters().dxdz();
              // dy/dz
              A(1,2) = updtsos.localParameters().dydz();
              // dx/dtheta_x
              A(0,3) = -updtsos.localPosition().y()*updtsos.localParameters().dxdz();
              // dy/dtheta_x
              A(1,3) = -updtsos.localPosition().y()*updtsos.localParameters().dydz();
              // dx/dtheta_y
              A(0,4) = -updtsos.localPosition().x()*updtsos.localParameters().dxdz();
              // dy/dtheta_y
              A(1,4) = -updtsos.localPosition().x()*updtsos.localParameters().dydz();
              // dx/dtheta_z
              A(0,5) = -updtsos.localPosition().y();
              // dy/dtheta_z
              A(1,5) = updtsos.localPosition().x();
              
              
  //             std::cout << "strip local z shift gradient: " << (Ralign*A.col(2))[0].value().value() << std::endl;
              
              // rotation from alignment basis to module local coordinates
  //             Matrix<AlignScalar, 2, 2> A;
  //             if (isglued) {
  //               const GlobalVector modx = preciseHit->det()->surface().toGlobal(LocalVector(1.,0.,0.));
  //               const GlobalVector mody = preciseHit->det()->surface().toGlobal(LocalVector(0.,1.,0.));
  //               
  //               const GlobalVector gluedx = parmDet->surface().toGlobal(LocalVector(1.,0.,0.));
  //               const GlobalVector gluedy = parmDet->surface().toGlobal(LocalVector(0.,1.,0.));
  //               
  //               A(0,0) = AlignScalar(modx.dot(gluedx));
  //               A(0,1) = AlignScalar(modx.dot(gluedy));
  //               A(1,0) = AlignScalar(mody.dot(gluedx));
  //               A(1,1) = AlignScalar(mody.dot(gluedy));
  //             }
  //             else {
  //               A = Matrix<AlignScalar, 2, 2>::Identity();
  //             }
  // 
  //             Matrix<AlignScalar, 2, 1> dh = dy0 - R*Hu*dx - R*A*dalpha;
              
                        
              double thetaincidence = std::asin(1./std::sqrt(std::pow(updtsos.localParameters().dxdz(),2) + std::pow(updtsos.localParameters().dydz(),2) + 1.));
              
  //             bool morehitquality = applyHitQuality_ ? thetaincidence > 0.25 : true;
              bool morehitquality = true;
              
              if (morehitquality) {
                nValidHitsFinal++;
                if (ispixel) {
                  nValidPixelHitsFinal++;
                }
              }
              else {
                Vinv = Matrix<AlignScalar, 2, 2>::Zero();
              }

              Matrix<AlignScalar, 2, 1> dh = dy0 - Ralign*Hu*dx - Ralign*A*dalpha;
              AlignScalar chisq = dh.transpose()*Vinv*dh;

              auto const& gradlocal = chisq.value().derivatives();
              //fill local hessian
              Matrix<double, nlocal, nlocal> hesslocal;
              for (unsigned int j=0; j<nlocal; ++j) {
                hesslocal.row(j) = chisq.derivatives()[j].derivatives();
              }
              
  //             Matrix<double, nlocal, 1> gradloctest0;
  //             Matrix<double, 1, 1> gradloctest1;
  //             Matrix<double, 2, 1> gradloctest2;
              
  //             std::cout << "nlocalalignment: " << nlocalalignment << " nlocal: " << nlocal << std::endl;
  //             std::cout << "gradlocal type: " << typeid(gradlocal).name() << std::endl;
  //             std::cout << "gradloctest0 type: " << typeid(gradloctest0).name() << std::endl;
  //             std::cout << "gradloctest1 type: " << typeid(gradloctest1).name() << std::endl;
  //             std::cout << "gradloctest2 type: " << typeid(gradloctest2).name() << std::endl;
  //             
  //             std::cout << "nhits: " << nhits << " nvalid: " << nvalid << " nvalidalign2d: " << nvalidalign2d << " ihit: " << ihit << std::endl;
  //             std::cout << "gradfull.size(): " << gradfull.size() << " nlocalstate: " << nlocalstate << " fullstateidx: " << fullstateidx << " nlocalparms: " << nlocalparms << " fullparmidx: " << fullparmidx << std::endl;
              
              // FIXME the templated block functions don't work here for some reason
              //fill global gradient
              gradfull.segment<nlocalstate>(fullstateidx) += gradlocal.head(nlocalstate);
              gradfull.segment<nlocalparms>(fullparmidx) += gradlocal.segment(localparmidx, nlocalparms);

              //fill global hessian (upper triangular blocks only)
              hessfull.block<nlocalstate,nlocalstate>(fullstateidx, fullstateidx) += hesslocal.topLeftCorner(nlocalstate,nlocalstate);
              hessfull.block<nlocalstate,nlocalparms>(fullstateidx, fullparmidx) += hesslocal.topRightCorner(nlocalstate, nlocalparms);
              hessfull.block<nlocalparms, nlocalparms>(fullparmidx, fullparmidx) += hesslocal.bottomRightCorner(nlocalparms, nlocalparms);
              
              for (unsigned int idim=0; idim<nlocalalignment; ++idim) {
                const unsigned int xglobalidx = detidparms.at(std::make_pair(alphaidxs[idim], preciseHit->geographicalId()));
                globalidxv[nparsBfield + nparsEloss + alignmentparmidx] = xglobalidx;
                alignmentparmidx++;
                if (alphaidxs[idim]==0) {
                  hitidxv.push_back(xglobalidx);
                }
              }
            };
            
            
            if (align2d) {
              fillAlignGrads(std::integral_constant<unsigned int, 3>());
            }
            else {
              fillAlignGrads(std::integral_constant<unsigned int, 2>());
            }
            
          }
          
        }

        if (!valid) {
          break;
        }
        
        trackstateidx += 3*tracknhits;
      }
      
      if (!valid) {
        continue;
      }
      
//       std::cout << nhits << std::endl;
//       std::cout << nvalid << std::endl;
//       std::cout << nvalidalign2d << std::endl;
//       std::cout << nparsAlignment << std::endl;
//       std::cout << alignmentparmidx << std::endl;
// 
//       std::cout << nparsBfield << std::endl;
//       std::cout << nparsEloss << std::endl;
//       std::cout << parmidx << std::endl;
      
      assert(parmidx == (nparsBfield + nparsEloss));
      assert(alignmentparmidx == nparsAlignment);
      
      //now do the expensive calculations and fill outputs
      
      //symmetrize the matrix (previous block operations do not guarantee that the needed blocks are filled)
      //TODO handle this more efficiently?
      hessfull.triangularView<StrictlyLower>() = hessfull.triangularView<StrictlyUpper>().transpose();
      
      auto const& dchisqdx = gradfull.head(nstateparms);
      auto const& dchisqdparms = gradfull.tail(npars);
      
      auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
      auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
      auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
      
      
      Cinvd.compute(d2chisqdx2);
      
      dxfull = -Cinvd.solve(dchisqdx);
      
      dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
      
  //     if (debugprintout_) {
  //       std::cout << "dxrefdparms" << std::endl;
  //       std::cout << dxdparms.leftCols<5>() << std::endl;
  //     }
      
      grad = dchisqdparms + dxdparms*dchisqdx;
      //TODO check the simplification
  //     hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
      hess = d2chisqdparms2 + dxdparms*d2chisqdxdparms;
      
      
      // compute final kinematics
      
      kinTree->movePointerToTheTop();
      RefCountedKinematicVertex dimu_vertex = kinTree->currentDecayVertex();
      
      // apply the GBL fit results to the vertex position
      Vector3d vtxpos;
      vtxpos << dimu_vertex->position().x(),
              dimu_vertex->position().y(),
              dimu_vertex->position().z();
      
      vtxpos += dxfull.head<3>();
      
      Jpsi_x = vtxpos[0];
      Jpsi_y = vtxpos[1];
      Jpsi_z = vtxpos[2];

      std::array<ROOT::Math::PxPyPzMVector, 2> muarr;
      std::array<Vector5d, 2> mucurvarr;
      std::array<int, 2> muchargearr;
      
      // apply the GBL fit results to the muon kinematics
      for (unsigned int id = 0; id < 2; ++id) {
        auto const &refFts = outparts[id]->currentState().freeTrajectoryState();
        auto const &jac = jacarr[id];
        
        JacobianCurvilinearToCartesian curv2cart(refFts.parameters());
        const AlgebraicMatrix65& jaccurv2cart = curv2cart.jacobian();
        const AlgebraicVector6 glob = refFts.parameters().vector();
        
        const Matrix<double, 5, 1> dxcurv = jac*dxfull;
        
        const Matrix<double, 6, 1> globupd = Map<const Matrix<double, 6, 1>>(glob.Array()) + Map<const Matrix<double, 6, 5, RowMajor>>(jaccurv2cart.Array())*dxcurv;
        
        double charge = std::copysign(1.0, refFts.charge()/refFts.momentum().mag() + dxcurv[0]);
        
        muarr[id] = ROOT::Math::PxPyPzMVector(globupd[3], globupd[4], globupd[5], mmu);
        muchargearr[id] = charge;
                
        auto &refParms = mucurvarr[id];
        CurvilinearTrajectoryParameters curvparms(refFts.position(), refFts.momentum(), refFts.charge());
        refParms << curvparms.Qbp(), curvparms.lambda(), curvparms.phi(), curvparms.xT(), curvparms.yT();
        refParms += dxcurv;

      }
      
      if ( (muchargearr[0] + muchargearr[1]) != 0) {
        continue;
      }

      const unsigned int idxplus = muchargearr[0] > 0 ? 0 : 1;
      const unsigned int idxminus = muchargearr[0] > 0 ? 1 : 0;
      
      Muplus_pt = muarr[idxplus].pt();
      Muplus_eta = muarr[idxplus].eta();
      Muplus_phi = muarr[idxplus].phi();
      
      Muminus_pt = muarr[idxminus].pt();
      Muminus_eta = muarr[idxminus].eta();
      Muminus_phi = muarr[idxminus].phi();
      
      Map<Matrix<float, 5, 1>>(Muplus_refParms.data()) = mucurvarr[idxplus].cast<float>();
      Map<Matrix<float, 5, 1>>(MuMinus_refParms.data()) = mucurvarr[idxminus].cast<float>();
      
      Muplus_jacRef.resize(5*npars);
      Map<Matrix<float, 5, Dynamic, RowMajor>>(Muplus_jacRef.data(), 5, npars) = (jacarr[idxplus]*dxdparms.transpose()).cast<float>();
      
      Muminus_jacRef.resize(5*npars);
      Map<Matrix<float, 5, Dynamic, RowMajor>>(Muminus_jacRef.data(), 5, npars) = (jacarr[idxminus]*dxdparms.transpose()).cast<float>();
      
      auto const jpsimom = muarr[0] + muarr[1];
      
      Jpsi_pt = jpsimom.pt();
      Jpsi_eta = jpsimom.eta();
      Jpsi_phi = jpsimom.phi();
      Jpsi_mass = jpsimom.mass();
      
      const reco::GenParticle *muplusgen = nullptr;
      const reco::GenParticle *muminusgen = nullptr;
      
      if (doGen_) {
        for (auto const &genpart : *genPartCollection) {
          if (genpart.status() != 1) {
            continue;
          }
          if (std::abs(genpart.pdgId()) != 13) {
            continue;
          }
          
          float dRplus = deltaR(genpart.phi(), muarr[idxplus].phi(), genpart.eta(), muarr[idxplus].eta());
          if (dRplus < 0.1 && genpart.charge() > 0) {
            muplusgen = &genpart;
          }
          
          float dRminus = deltaR(genpart.phi(), muarr[idxminus].phi(), genpart.eta(), muarr[idxminus].eta());
          if (dRminus < 0.1 && genpart.charge() < 0) {
            muminusgen = &genpart;
          }
        }
      }
      
      if (muplusgen != nullptr) {
        Muplusgen_pt = muplusgen->pt();
        Muplusgen_eta = muplusgen->eta();
        Muplusgen_phi = muplusgen->phi();
      }
      else {
        Muplusgen_pt = -99.;
        Muplusgen_eta = -99.;
        Muplusgen_phi = -99.;
      }
      
      if (muminusgen != nullptr) {
        Muminusgen_pt = muminusgen->pt();
        Muminusgen_eta = muminusgen->eta();
        Muminusgen_phi = muminusgen->phi();
      }
      else {
        Muminusgen_pt = -99.;
        Muminusgen_eta = -99.;
        Muminusgen_phi = -99.;
      }
      
      if (muplusgen != nullptr && muminusgen != nullptr) {
        auto const jpsigen = ROOT::Math::PtEtaPhiMVector(muplusgen->pt(), muplusgen->eta(), muplusgen->phi(), mmu) +
                            ROOT::Math::PtEtaPhiMVector(muminusgen->pt(), muminusgen->eta(), muminusgen->phi(), mmu);
        
        Jpsigen_pt = jpsigen.pt();
        Jpsigen_eta = jpsigen.eta();
        Jpsigen_phi = jpsigen.phi();
        Jpsigen_mass = jpsigen.mass();
        
        Jpsigen_x = muplusgen->vx();
        Jpsigen_y = muplusgen->vy();
        Jpsigen_z = muplusgen->vz();
      }
      else {
        Jpsigen_pt = -99.;
        Jpsigen_eta = -99.;
        Jpsigen_phi = -99.;
        Jpsigen_mass = -99.;
        
        Jpsigen_x = -99.;
        Jpsigen_y = -99.;
        Jpsigen_z = -99.;
      }
      
      
  //     const Vector5d dxRef = dxfull.head<5>();
  //     const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();


      nParms = npars;

      gradv.clear();
      gradv.resize(npars,0.);
      
      if (fillTrackTree_ && fillGrads_) {
        tree->SetBranchAddress("gradv", gradv.data());
      }
      
      //eigen representation of the underlying vector storage
      Map<VectorXf> gradout(gradv.data(), npars);

      gradout = grad.cast<float>();
      
        

      gradmax = 0.;
      for (unsigned int i=0; i<npars; ++i) {
        const float absval = std::abs(grad[i]);
        if (absval>gradmax) {
          gradmax = absval;
        }      
      }
      
      
      hessmax = 0.;
      for (unsigned int i=0; i<npars; ++i) {
        for (unsigned int j=i; j<npars; ++j) {
          const unsigned int iidx = globalidxv[i];
          const unsigned int jidx = globalidxv[j];
          
          const float absval = std::abs(hess(i,j));
          if (absval>hessmax) {
            hessmax = absval;
          }
          
        }
        
      }
        
      //fill packed hessian and indices
      const unsigned int nsym = npars*(1+npars)/2;
      hesspackedv.clear();    
      hesspackedv.resize(nsym, 0.);
      
      nSym = nsym;
      if (fillTrackTree_ && fillGrads_) {
        tree->SetBranchAddress("hesspackedv", hesspackedv.data());
      }
      
      Map<VectorXf> hesspacked(hesspackedv.data(), nsym);
      const Map<const VectorXu> globalidx(globalidxv.data(), npars);

      unsigned int packedidx = 0;
      for (unsigned int ipar = 0; ipar < npars; ++ipar) {
        const unsigned int segmentsize = npars - ipar;
        hesspacked.segment(packedidx, segmentsize) = hess.block<1, Dynamic>(ipar, ipar, 1, segmentsize).cast<float>();
        packedidx += segmentsize;
      }

      if (fillTrackTree_) {
        tree->Fill();
      }
      

      
//       const Matrix3d covvtx = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<3,3>();
      
//       const VectorXd cinvrow0 = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).row(0).head(nstateparms);
      
//       std::cout << "kinfit covariance:" << std::endl;
//       std::cout << dimu_vertex->error().matrix() << std::endl;
//       
//       std::cout << "GBL covariance:" << std::endl;
//       std::cout << 2.*covvtx << std::endl;
//       
//       std::cout << "cinvrow0" << std::endl;
//       std::cout << cinvrow0 << std::endl;
      
      //TODO restore statejac stuff
//       dxstate = statejac*dxfull;
//       const Vector5d dxRef = dxstate.head<5>();
//       const Matrix5d Cinner = (statejac*Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))*statejac.transpose()).topLeftCorner<5,5>();
      
      //TODO fill outputs
      
    }
  }
  
}


DEFINE_FWK_MODULE(ResidualGlobalCorrectionMakerTwoTrack);