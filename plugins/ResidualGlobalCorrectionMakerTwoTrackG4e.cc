#include "ResidualGlobalCorrectionMakerBase.h"
#include "MagneticFieldOffset.h"

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

#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"


class ResidualGlobalCorrectionMakerTwoTrackG4e : public ResidualGlobalCorrectionMakerBase
{
public:
  explicit ResidualGlobalCorrectionMakerTwoTrackG4e(const edm::ParameterSet &);
  ~ResidualGlobalCorrectionMakerTwoTrackG4e() {}

//   static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  
  Matrix<double, 1, 6> massJacobian(const FreeTrajectoryState &state0, const FreeTrajectoryState &state1, double dmass) const;
  
  virtual void beginStream(edm::StreamID) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  
  bool doMassConstraint_;
  double massConstraint_;
  double massConstraintWidth_;
  
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
  
  float Jpsikin_x;
  float Jpsikin_y;
  float Jpsikin_z;
  float Jpsikin_pt;
  float Jpsikin_eta;
  float Jpsikin_phi;
  float Jpsikin_mass;
  
  float Mupluskin_pt;
  float Mupluskin_eta;
  float Mupluskin_phi;
  
  float Muminuskin_pt;
  float Muminuskin_eta;
  float Muminuskin_phi;
  
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
  
  std::array<float, 3> Muplus_refParms;
  std::array<float, 3> MuMinus_refParms;
  
  std::vector<float> Muplus_jacRef;
  std::vector<float> Muminus_jacRef;
  
  unsigned int Muplus_nhits;
  unsigned int Muplus_nvalid;
  unsigned int Muplus_nvalidpixel;
  
  unsigned int Muminus_nhits;
  unsigned int Muminus_nvalid;
  unsigned int Muminus_nvalidpixel;
  
//   std::vector<float> hessv;
  

  
};


ResidualGlobalCorrectionMakerTwoTrackG4e::ResidualGlobalCorrectionMakerTwoTrackG4e(const edm::ParameterSet &iConfig) : ResidualGlobalCorrectionMakerBase(iConfig) 
{
  doMassConstraint_ = iConfig.getParameter<bool>("doMassConstraint");
  massConstraint_ = iConfig.getParameter<double>("massConstraint");
  massConstraintWidth_ = iConfig.getParameter<double>("massConstraintWidth");
}

void ResidualGlobalCorrectionMakerTwoTrackG4e::beginStream(edm::StreamID streamid)
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
    
    tree->Branch("Jpsikin_x", &Jpsikin_x);
    tree->Branch("Jpsikin_y", &Jpsikin_y);
    tree->Branch("Jpsikin_z", &Jpsikin_z);
    tree->Branch("Jpsikin_pt", &Jpsikin_pt);
    tree->Branch("Jpsikin_eta", &Jpsikin_eta);
    tree->Branch("Jpsikin_phi", &Jpsikin_phi);
    tree->Branch("Jpsikin_mass", &Jpsikin_mass);
    
    tree->Branch("Mupluskin_pt", &Mupluskin_pt);
    tree->Branch("Mupluskin_eta", &Mupluskin_eta);
    tree->Branch("Mupluskin_phi", &Mupluskin_phi);
    
    tree->Branch("Muminuskin_pt", &Muminuskin_pt);
    tree->Branch("Muminuskin_eta", &Muminuskin_eta);
    tree->Branch("Muminuskin_phi", &Muminuskin_phi);
    
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
    
    tree->Branch("Muplus_refParms", Muplus_refParms.data(), "Muplus_refParms[3]/F");
    tree->Branch("MuMinus_refParms", MuMinus_refParms.data(), "MuMinus_refParms[3]/F");
    
    tree->Branch("Muplus_jacRef", &Muplus_jacRef);
    tree->Branch("Muminus_jacRef", &Muminus_jacRef);
    
    tree->Branch("Muplus_nhits", &Muplus_nhits);
    tree->Branch("Muplus_nvalid", &Muplus_nvalid);
    tree->Branch("Muplus_nvalidpixel", &Muplus_nvalidpixel);
    
    tree->Branch("Muminus_nhits", &Muminus_nhits);
    tree->Branch("Muminus_nvalid", &Muminus_nvalid);
    tree->Branch("Muminus_nvalidpixel", &Muminus_nvalidpixel);
  
//     tree->Branch("hessv", &hessv);
    
  }
}


// ------------ method called for each event  ------------
void ResidualGlobalCorrectionMakerTwoTrackG4e::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
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
  iSetup.get<TrackingComponentsRecord>().get("Geant4ePropagator", thePropagator);

  const MagneticField* field = thePropagator->magneticField();
  const Geant4ePropagator *g4prop = dynamic_cast<const Geant4ePropagator*>(thePropagator.product());
  
  Handle<std::vector<reco::GenParticle>> genPartCollection;
  if (doGen_) {
    iEvent.getByToken(GenParticlesToken_, genPartCollection);
  }
  
  KFUpdator updator;
  TkClonerImpl const& cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(ttrh.product())->cloner();
  
  edm::ESHandle<TransientTrackBuilder> TTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", TTBuilder);
  KinematicParticleFactoryFromTransientTrack pFactory;

  Handle<reco::BeamSpot> bsH;
  iEvent.getByToken(inputBs_, bsH);
  
  constexpr double mmu = 0.1056583745;
  constexpr double mmuerr = 0.0000000024;

  VectorXd gradfull;
  MatrixXd hessfull;
  
  VectorXd dxfull;
  MatrixXd dxdparms;
  VectorXd grad;
  MatrixXd hess;
  LDLT<MatrixXd> Cinvd;
//   FullPivLU<MatrixXd> Cinvd;
//   ColPivHouseholderQR<MatrixXd> Cinvd;
  
  std::array<MatrixXd, 2> jacarr;
  
  
  // loop over combinatorics of track pairs
  for (auto itrack = trackOrigH->begin(); itrack != trackOrigH->end(); ++itrack) {
    const reco::TransientTrack itt = TTBuilder->build(*itrack);
    for (auto jtrack = itrack + 1; jtrack != trackOrigH->end(); ++jtrack) {
      const reco::TransientTrack jtt = TTBuilder->build(*jtrack);
      
      
      const reco::GenParticle *mu0gen = nullptr;
      const reco::GenParticle *mu1gen = nullptr;
      
      double massconstraintval = massConstraint_;
      if (doGen_) {
        for (auto const &genpart : *genPartCollection) {
          if (genpart.status() != 1) {
            continue;
          }
          if (std::abs(genpart.pdgId()) != 13) {
            continue;
          }
          
          float dR0 = deltaR(genpart.phi(), itrack->phi(), genpart.eta(), itrack->eta());
          if (dR0 < 0.1 && genpart.charge() == itrack->charge()) {
            mu0gen = &genpart;
          }
          
          float dR1 = deltaR(genpart.phi(), jtrack->phi(), genpart.eta(), jtrack->eta());
          if (dR1 < 0.1 && genpart.charge() == jtrack->charge()) {
            mu1gen = &genpart;
          }
        }
        
//         if (mu0gen != nullptr && mu1gen != nullptr) {
//           auto const jpsigen = ROOT::Math::PtEtaPhiMVector(mu0gen->pt(), mu0gen->eta(), mu0gen->phi(), mmu) +
//                             ROOT::Math::PtEtaPhiMVector(mu1gen->pt(), mu1gen->eta(), mu1gen->phi(), mmu);
// 
//           massconstraintval = jpsigen.mass();
//         }
//         else {
//           continue;
//         }
        
      }
      
//       std::cout << "massconstraintval = " << massconstraintval << std::endl;
      
      // common vertex fit
      std::vector<RefCountedKinematicParticle> parts;
      
      float masserr = mmuerr;
      float chisq = 0.;
      float ndf = 0.;
      parts.push_back(pFactory.particle(itt, mmu, chisq, ndf, masserr));
      parts.push_back(pFactory.particle(jtt, mmu, chisq, ndf, masserr));
      
      RefCountedKinematicTree kinTree;
      if (doMassConstraint_) {
//       if (false) {
//         TwoTrackMassKinematicConstraint constraint(massConstraint_);
        TwoTrackMassKinematicConstraint constraint(massconstraintval);
        KinematicConstrainedVertexFitter vtxFitter;
        kinTree = vtxFitter.fit(parts, &constraint);
      }
      else {
        KinematicParticleVertexFitter vtxFitter;
        kinTree = vtxFitter.fit(parts);
      }
      
      if (kinTree->isEmpty() || !kinTree->isConsistent()) {
        continue;
      }
      
      kinTree->movePointerToTheTop();
      RefCountedKinematicParticle dimu_kinfit = kinTree->currentParticle();
      const double m0 = dimu_kinfit->currentState().mass();
      
      if (false) {
        // debug output
//         kinTree->movePointerToTheTop();
        
//         RefCountedKinematicParticle dimu_kinfit = kinTree->currentParticle();
        RefCountedKinematicVertex dimu_vertex = kinTree->currentDecayVertex();
        
        std::cout << dimu_kinfit->currentState().mass() << std::endl;
        std::cout << dimu_vertex->position() << std::endl;
      }
      
      const std::vector<RefCountedKinematicParticle> outparts = kinTree->finalStateParticles();
      std::array<FreeTrajectoryState, 2> refftsarr = {{ outparts[0]->currentState().freeTrajectoryState(),
                                                        outparts[1]->currentState().freeTrajectoryState() }};
                                                        
      if (fitFromGenParms_ && mu0gen != nullptr && mu1gen != nullptr) {
        GlobalPoint pos0(mu0gen->vertex().x(), mu0gen->vertex().y(), mu0gen->vertex().z());
        GlobalVector mom0(mu0gen->momentum().x(), mu0gen->momentum().y(), mu0gen->momentum().z());
        
        GlobalPoint pos1(mu1gen->vertex().x(), mu1gen->vertex().y(), mu1gen->vertex().z());
        GlobalVector mom1(mu1gen->momentum().x(), mu1gen->momentum().y(), mu1gen->momentum().z());
        
        refftsarr[0] = FreeTrajectoryState(pos0, mom0, mu0gen->charge(), field);
        refftsarr[1] = FreeTrajectoryState(pos1, mom1, mu1gen->charge(), field);
      }
      
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
                
//                 std::cout << "getSplitClusterErrorX = " << cluster.getSplitClusterErrorX() << std::endl;
                
//                 const double jpsieta = dimu_kinfit->currentState().freeTrajectoryState().momentum().eta();
//                 const double jpsipt = dimu_kinfit->currentState().freeTrajectoryState().momentum().perp();
//                 if (std::abs(jpsieta)>2. && jpsipt>20. && it == track.recHitsBegin()) {
//                   std::cout << "id = " << id << " detid = " << (*it)->geographicalId().rawId() << " minPixelRow = " << cluster.minPixelRow() << " maxPixelRow = " << cluster.maxPixelRow() << " minPixelCol = " << cluster.minPixelCol() << " maxPixelCol = " << cluster.maxPixelCol() << std::endl;
//                 }
                
                hitquality = !pixhit->isOnEdge() && cluster.sizeX() > 1;
              }
              else {
                assert(tkhit->cluster_strip().isNonnull());
                const SiStripCluster& cluster = *tkhit->cluster_strip();
                const StripTopology* striptopology = dynamic_cast<const StripTopology*>(&(detectorG->topology()));
                assert(striptopology);
                
                const uint16_t firstStrip = cluster.firstStrip();
                const uint16_t lastStrip = cluster.firstStrip() + cluster.amplitudes().size() - 1;
                const bool isOnEdge = firstStrip == 0 || lastStrip == (striptopology->nstrips() - 1);
                
    //             if (isOnEdge) {
    //               std::cout << "strip hit isOnEdge" << std::endl;
    //             }
                
//                 hitquality = !isOnEdge;
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
      
      std::array<std::vector<TrajectoryStateOnSurface>, 2> layerStatesarr;
      
      std::array<unsigned int, 2> nhitsarr = {{ 0, 0 }};
      std::array<unsigned int, 2> nvalidarr = {{ 0, 0 }};
      std::array<unsigned int, 2> nvalidpixelarr = {{ 0, 0 }};
      
      // second loop to count hits
      for (unsigned int id = 0; id < 2; ++id) {
        auto const &hits = hitsarr[id];
        layerStatesarr[id].reserve(hits.size());
        for (auto const &hit : hits) {
          ++nhits;
          ++nhitsarr[id];
          if (hit->isValid()) {
            ++nvalid;
            ++nvalidarr[id];
            
            const bool ispixel = GeomDetEnumerators::isTrackerPixel(hit->det()->subDetector());
            if (ispixel) {
              ++nvalidpixel;
              ++nvalidpixelarr[id];
            }
            
            
            const bool align2d = detidparms.count(std::make_pair(1, hit->geographicalId()));
            if (align2d) {
              ++nvalidalign2d;
            }
          } 
        }
      }
      
      AlgebraicSymMatrix55 null55;
      const CurvilinearTrajectoryError nullerr(null55);

      
      const unsigned int nparsAlignment = 2*nvalid + nvalidalign2d;
      const unsigned int nparsBfield = nhits;
      const unsigned int nparsEloss = nhits;
      const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
      
      const unsigned int nstateparms =  9 + 5*nhits;
      const unsigned int nparmsfull = nstateparms + npars;
      
      bool valid = true;
      
//       constexpr unsigned int niters = 1;
      constexpr unsigned int niters = 3;
//       constexpr unsigned int niters = 10;
      
      for (unsigned int iiter=0; iiter<niters; ++iiter) {
        
        gradfull = VectorXd::Zero(nparmsfull);
        hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);

        globalidxv.clear();
        globalidxv.resize(npars, 0);
        
        nParms = npars;
        if (fillTrackTree_) {
          tree->SetBranchAddress("globalidxv", globalidxv.data());
        }
        
        std::array<Matrix<double, 5, 7>, 2> FdFmrefarr;
        std::array<unsigned int, 2> trackstateidxarr;
        std::array<unsigned int, 2> trackparmidxarr;
        
        unsigned int trackstateidx = 3;
        unsigned int parmidx = 0;
        unsigned int alignmentparmidx = 0;
        
        double chisq0val = 0.;
        
        
//         const bool firsthitshared = hitsarr[0][0]->sharesInput(&(*hitsarr[1][0]), TrackingRecHit::some);
        
//         std::cout << "firsthitshared = " << firsthitshared << std::endl;
        
        for (unsigned int id = 0; id < 2; ++id) {
  //         FreeTrajectoryState refFts = outparts[id]->currentState().freeTrajectoryState();
          FreeTrajectoryState &refFts = refftsarr[id];
          auto &hits = hitsarr[id];
          
          std::vector<TrajectoryStateOnSurface> &layerStates = layerStatesarr[id];
                    
          trackstateidxarr[id] = trackstateidx;
          trackparmidxarr[id] = parmidx;
          
          const unsigned int tracknhits = hits.size();
          
          if (iiter > 0) {
            //update current state from reference point state (errors not needed beyond first iteration)
            JacobianCurvilinearToCartesian curv2cart(refFts.parameters());
            const AlgebraicMatrix65& jac = curv2cart.jacobian();
            const AlgebraicVector6 glob = refFts.parameters().vector();
            
            const Matrix<double, 3, 1> posupd = Map<const Matrix<double, 6, 1>>(glob.Array()).head<3>() + dxfull.head<3>();
            
            const Matrix<double, 3, 1> momupd = Map<const Matrix<double, 6, 1>>(glob.Array()).tail<3>() + Map<const Matrix<double, 6, 5, RowMajor>>(jac.Array()).bottomLeftCorner<3, 3>()*dxfull.segment<3>(trackstateidx);
            
            const GlobalPoint pos(posupd[0], posupd[1], posupd[2]);
            const GlobalVector mom(momupd[0], momupd[1], momupd[2]);
            const double charge = std::copysign(1., refFts.charge()/refFts.momentum().mag() + dxfull[trackstateidx]);
      //         std::cout << "before update: reffts:" << std::endl;
      //         std::cout << refFts.parameters().vector() << std::endl;
      //         std::cout << "charge " << refFts.charge() << std::endl;
            refFts = FreeTrajectoryState(pos, mom, charge, field);
      //         std::cout << "after update: reffts:" << std::endl;
      //         std::cout << refFts.parameters().vector() << std::endl;
      //         std::cout << "charge " << refFts.charge() << std::endl;
      //         currentFts = refFts;
          }
          
          // initialize with zero uncertainty
          refFts = FreeTrajectoryState(refFts.parameters(), nullerr);
          
//           std::cout << "refFts:" << std::endl;
//           std::cout << refFts.position() << std::endl;
//           std::cout << refFts.momentum() << std::endl;
//           std::cout << refFts.charge() << std::endl;
          
//           auto const &surface0 = *hits[0]->surface();
          auto const &surface0 = *surfacemap_.at(hits[0]->geographicalId());
//           auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, surface0);
    //       auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, *beampipe);
          auto const propresultref = g4prop->propagateGenericWithJacobian(refFts, surface0);
          if (!propresultref.first.isValid()) {
            std::cout << "Abort: Propagation of reference state Failed!" << std::endl;
            valid = false;
            break;
          }
          TrajectoryStateOnSurface updtsos = propresultref.first;
          
          const Matrix<double, 5, 6> hybrid2curvref = hybrid2curvJacobian(refFts);
          
//           JacobianCartesianToCurvilinear cart2curvref(refFts.parameters());
//           auto const &jacCart2CurvRef = Map<const Matrix<double, 5, 6, RowMajor>>(cart2curvref.jacobian().Array());
          
          Matrix<double, 5, 7> FdFm = Map<const Matrix<double, 5, 7, RowMajor>>(propresultref.second.Array());
          
          FdFmrefarr[id] = FdFm;
          
          if (bsConstraint_) {
            // apply beamspot constraint
            // TODO add residual corrections for beamspot parameters?
            
            constexpr unsigned int nlocalvtx = 3;
            
            constexpr unsigned int nlocal = nlocalvtx;
            
            constexpr unsigned int localvtxidx = 0;
            
            constexpr unsigned int fullvtxidx = 0;
            
            using BSScalar = AANT<double, nlocal>;
            
            const double sigb1 = bsH->BeamWidthX();
            const double sigb2 = bsH->BeamWidthY();
            const double sigb3 = bsH->sigmaZ();
            const double dxdz = bsH->dxdz();
            const double dydz = bsH->dydz();
            const double x0 = bsH->x0();
            const double y0 = bsH->y0();
            const double z0 = bsH->z0();
            
            
            // covariance matrix of luminous region in global coordinates
            // taken from https://github.com/cms-sw/cmssw/blob/abc1f17b230effd629c9565fb5d95e527abcb294/RecoVertex/BeamSpotProducer/src/FcnBeamSpotFitPV.cc#L63-L90

            // FIXME xy correlation is not stored and assumed to be zero
            const double corrb12 = 0.;
            
            const double varb1 = sigb1*sigb1;
            const double varb2 = sigb2*sigb2;
            const double varb3 = sigb3*sigb3;
            
            Matrix<double, 3, 3> covBS = Matrix<double, 3, 3>::Zero();
            // parametrisation: rotation (dx/dz, dy/dz); covxy
            covBS(0,0) = varb1;
            covBS(1,0) = covBS(0,1) = corrb12*sigb1*sigb2;
            covBS(1,1) = varb2;
            covBS(2,0) = covBS(0,2) = dxdz*(varb3-varb1)-dydz*covBS(1,0);
            covBS(2,1) = covBS(1,2) = dydz*(varb3-varb2)-dxdz*covBS(1,0);
            covBS(2,2) = varb3;

    //         std::cout << "covBS:" << std::endl;
    //         std::cout << covBS << std::endl;
            
            Matrix<BSScalar, 3, 1> dvtx = Matrix<BSScalar, 3, 1>::Zero();
            for (unsigned int j=0; j<dvtx.size(); ++j) {
              init_twice_active_var(dvtx[j], nlocal, localvtxidx + j);
            }
            
            Matrix<BSScalar, 3, 1> dbs0;
            dbs0[0] = BSScalar(refFts.position().x() - x0);
            dbs0[1] = BSScalar(refFts.position().y() - y0);
            dbs0[2] = BSScalar(refFts.position().z() - z0);
            
    //         std::cout << "dposition / d(qop, lambda, phi) (should be 0?):" << std::endl;
    //         std::cout << Map<const Matrix<double, 6, 5, RowMajor>>(jac.Array()).topLeftCorner<3,3>() << std::endl;
            
            const Matrix<BSScalar, 3, 3> covBSinv = covBS.inverse().cast<BSScalar>();
            
            const Matrix<BSScalar, 3, 1> dbs = dbs0 + dvtx;
            const BSScalar chisq = dbs.transpose()*covBSinv*dbs;
            
            chisq0val += chisq.value().value();
            
            auto const& gradlocal = chisq.value().derivatives();
            //fill local hessian
            Matrix<double, nlocal, nlocal> hesslocal;
            for (unsigned int j=0; j<nlocal; ++j) {
              hesslocal.row(j) = chisq.derivatives()[j].derivatives();
            }
            
            //fill global gradient
            gradfull.segment<nlocalvtx>(fullvtxidx) += gradlocal.head<nlocalvtx>();
            //fill global hessian (upper triangular blocks only)
            hessfull.block<nlocalvtx,nlocalvtx>(fullvtxidx, fullvtxidx) += hesslocal.topLeftCorner<nlocalvtx,nlocalvtx>();
            
          }
          
          

          for (unsigned int ihit = 0; ihit < hits.size(); ++ihit) {
    //         std::cout << "ihit " << ihit << std::endl;
            auto const& hit = hits[ihit];
            
            const uint32_t gluedid = trackerTopology->glued(hit->det()->geographicalId());
            const bool isglued = gluedid != 0;
            const DetId parmdetid = isglued ? DetId(gluedid) : hit->geographicalId();
            const GeomDet* parmDet = isglued ? globalGeometry->idToDet(parmdetid) : hit->det();
            const double xifraction = isglued ? hit->det()->surface().mediumProperties().xi()/parmDet->surface().mediumProperties().xi() : 1.;

            if (ihit > 0) {
              if (std::abs(updtsos.globalMomentum().eta()) > 4.0) {
                std::cout << "WARNING:  Invalid state!!!" << std::endl;
                valid = false;
                break;
              }
              
              
    //        auto const &surfaceip1 = *hits[ihit+1]->surface();
    //           auto const &surface = *hit->surface();
    //           const Plane &surface = *hit->surface();
              auto const &surface = *surfacemap_.at(hit->geographicalId());
    //           auto propresult = thePropagator->propagateWithPath(updtsos, surface);
              auto const propresult = g4prop->propagateGenericWithJacobian(*updtsos.freeState(), surface);
    //           propresult = fPropagator->geometricalPropagator().propagateWithPath(updtsos, *hits[ihit+1]->surface());
              if (!propresult.first.isValid()) {
                std::cout << "Abort: Propagation Failed!" << std::endl;
                valid = false;
                break;
              }
              
              FdFm = Map<const Matrix<double, 5, 7, RowMajor>>(propresult.second.Array());
    //           FdFm = localTransportJacobian(updtsos, propresult, false);
              updtsos = propresult.first;
            }
                        


            
            // compute convolution correction in local coordinates (BEFORE material effects are applied)
    //         const Matrix<double, 2, 1> dxlocalconv = localPositionConvolution(updtsos);
            
            // curvilinear to local jacobian
            JacobianCurvilinearToLocal curv2localm(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
            const AlgebraicMatrix55& curv2localjacm = curv2localm.jacobian();
            const Matrix<double, 5, 5> Hm = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacm.Array()); 
 
            //get the process noise matrix
            AlgebraicMatrix55 const Qmat = updtsos.localError().matrix();
            const Map<const Matrix<double, 5, 5, RowMajor>>Q(Qmat.Array());

            // update state from previous iteration
            //momentum kink residual
            AlgebraicVector5 idx0(0., 0., 0., 0., 0.);
            if (iiter==0) {
              updtsos.update(updtsos.localParameters(),
                              LocalTrajectoryError(0.,0.,0.,0.,0.),
                              updtsos.surface(),
                              updtsos.magneticField(),
                              updtsos.surfaceSide());
              layerStates.push_back(updtsos);
            }
            else {          
              //current state from previous state on this layer
              //save current parameters          
              TrajectoryStateOnSurface& oldtsos = layerStates[ihit];
              
              JacobianCurvilinearToLocal curv2localold(oldtsos.surface(), oldtsos.localParameters(), *oldtsos.magneticField());
              const AlgebraicMatrix55& curv2localjacold = curv2localold.jacobian();
              const Matrix<double, 5, 5> Hold = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacold.Array()); 
              
              const AlgebraicVector5 local = oldtsos.localParameters().vector();
              
              auto const& dxlocal = Hold*dxfull.segment<5>(trackstateidx + 3 + 5*ihit);
              const Matrix<double, 5, 1> localupd = Map<const Matrix<double, 5, 1>>(local.Array()) + dxlocal;
              AlgebraicVector5 localvecupd(localupd[0],localupd[1],localupd[2],localupd[3],localupd[4]);
              
              idx0 = localvecupd - updtsos.localParameters().vector();
              
              const LocalTrajectoryParameters localparms(localvecupd, oldtsos.localParameters().pzSign());
              
    //           std::cout << "before update: oldtsos:" << std::endl;
    //           std::cout << oldtsos.localParameters().vector() << std::endl;
//               oldtsos.update(localparms, oldtsos.surface(), field, oldtsos.surfaceSide());
              oldtsos.update(localparms, LocalTrajectoryError(0.,0.,0.,0.,0.), oldtsos.surface(), field, oldtsos.surfaceSide());
    //           std::cout << "after update: oldtsos:" << std::endl;
    //           std::cout << oldtsos.localParameters().vector() << std::endl;
              updtsos = oldtsos;

            }
            
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
            
            // curvilinear to local jacobian
            JacobianCurvilinearToLocal curv2localp(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
            const AlgebraicMatrix55& curv2localjacp = curv2localp.jacobian();
            const Matrix<double, 5, 5> Hp = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacp.Array());
            
//             const Matrix<double, 5, 5> Hpalt = curv2localJacobianAlt(updtsos);
//             
//             std::cout << "Hp" << std::endl;
//             std::cout << Hp << std::endl;
//             std::cout << "Hpalt" << std::endl;
//             std::cout << Hpalt << std::endl;
            
            
            if (true) {
              
  //             std::cout << "EdE first hit:" << std::endl;
  //             std::cout << EdE << std::endl;
  //             
  //             std::cout << "xival = " << xival << std::endl;
              
//               AlgebraicVector5 idx0(0., 0., 0., 0., 0.);
              const Vector5d dx0 = Map<const Vector5d>(idx0.Array());

              if (ihit == 0) {
                constexpr unsigned int nvtxstate = 3;
                constexpr unsigned int nlocalstate = 8;
                constexpr unsigned int nlocalbfield = 1;
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
                                
                Matrix<MSScalar, 5, 3> Fvtx = (FdFm.leftCols<5>()*hybrid2curvref.rightCols<3>()).cast<MSScalar>();
                Matrix<MSScalar, 5, 3> Fmom = (FdFm.leftCols<5>()*hybrid2curvref.leftCols<3>()).cast<MSScalar>();
                
//                 Matrix<MSScalar, 5, 3> Fvtx = (FdFm.leftCols<5>()*jacCart2CurvRef.leftCols<3>()).cast<MSScalar>();
//                 Matrix<MSScalar, 5, 3> Fmom = FdFm.leftCols<3>().cast<MSScalar>();
                
                Matrix<MSScalar, 5, 1> Fb = FdFm.col(5).cast<MSScalar>();
                Matrix<MSScalar, 5, 1> Fxi = FdFm.col(6).cast<MSScalar>();
                
                Matrix<MSScalar, 5, 5> Hmstate = Hm.cast<MSScalar>();
                Matrix<MSScalar, 5, 5> Hpstate = Hp.cast<MSScalar>();
                
                Matrix<MSScalar, 5, 5> Qinv = Q.inverse().cast<MSScalar>();
                
                // initialize active scalars for common vertex parameters
                Matrix<MSScalar, 3, 1> dvtx = Matrix<MSScalar, 3, 1>::Zero();
                for (unsigned int j=0; j<dvtx.size(); ++j) {
                  init_twice_active_var(dvtx[j], nlocal, localvtxidx + j);
                }
                
                Matrix<MSScalar, 3, 1> dmom = Matrix<MSScalar, 3, 1>::Zero();
                for (unsigned int j=0; j<dmom.size(); ++j) {
                  init_twice_active_var(dmom[j], nlocal, localstateidx + j);
                }
                
                Matrix<MSScalar, 5, 1> du = Matrix<MSScalar, 5, 1>::Zero();
                for (unsigned int j=0; j<du.size(); ++j) {
                  init_twice_active_var(du[j], nlocal, localstateidx + 3 + j);
                }
                
                // initialize active scalars for correction parameters

                MSScalar dbeta(0.);
                init_twice_active_var(dbeta, nlocal, localparmidx);
                
                MSScalar dxi(0.);
                init_twice_active_var(dxi, nlocal, localparmidx + 1);
                
                const Matrix<MSScalar, 5, 1> dprop = dx0.cast<MSScalar>() + Hpstate*du - Hmstate*Fvtx*dvtx - Hmstate*Fmom*dmom - Hmstate*Fb*dbeta - Hmstate*Fxi*dxi;
                const MSScalar chisq = dprop.transpose()*Qinv*dprop;
                
                chisq0val += chisq.value().value();
                
                auto const& gradlocal = chisq.value().derivatives();
                //fill local hessian
                Matrix<double, nlocal, nlocal> hesslocal;
                for (unsigned int j=0; j<nlocal; ++j) {
                  hesslocal.row(j) = chisq.derivatives()[j].derivatives();
                }
                
                constexpr std::array<unsigned int, 3> localsizes = {{ nvtxstate, nlocalstate, nlocalparms }};
                constexpr std::array<unsigned int, 3> localidxs = {{ localvtxidx, localstateidx, localparmidx }};
                const std::array<unsigned int, 3> fullidxs = {{ fullvtxidx, fullstateidx, fullparmidx }};
                
                for (unsigned int iidx = 0; iidx < localidxs.size(); ++iidx) {
                  gradfull.segment(fullidxs[iidx], localsizes[iidx]) += gradlocal.segment(localidxs[iidx], localsizes[iidx]);
                  for (unsigned int jidx = 0; jidx < localidxs.size(); ++jidx) {
                    hessfull.block(fullidxs[iidx], fullidxs[jidx], localsizes[iidx], localsizes[jidx]) += hesslocal.block(localidxs[iidx], localidxs[jidx], localsizes[iidx], localsizes[jidx]);
                  }
                }
                                
              }
              else {
                //TODO statejac stuff
                
                constexpr unsigned int nlocalstate = 10;
                constexpr unsigned int nlocalbfield = 1;
                constexpr unsigned int nlocaleloss = 1;
                constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
                
                constexpr unsigned int nlocal = nlocalstate + nlocalparms;
                
                constexpr unsigned int localstateidx = 0;
                constexpr unsigned int localparmidx = localstateidx + nlocalstate;
                
                const unsigned int fullstateidx = trackstateidx + 3 + 5*(ihit - 1);
                const unsigned int fullparmidx = nstateparms + parmidx;
                
                using MSScalar = AANT<double, nlocal>;

                Matrix<MSScalar, 5, 5> Fstate = FdFm.leftCols<5>().cast<MSScalar>();
                Matrix<MSScalar, 5, 1> Fb = FdFm.col(5).cast<MSScalar>();
                Matrix<MSScalar, 5, 1> Fxi = FdFm.col(6).cast<MSScalar>();
                
                Matrix<MSScalar, 5, 5> Hmstate = Hm.cast<MSScalar>();
                Matrix<MSScalar, 5, 5> Hpstate = Hp.cast<MSScalar>();
                
                Matrix<MSScalar, 5, 5> Qinv = Q.inverse().cast<MSScalar>();
                                        
                // initialize active scalars for state parameters
                Matrix<MSScalar, 5, 1> dum = Matrix<MSScalar, 5, 1>::Zero();
                //suppress gradients of reference point parameters when fitting with gen constraint
                for (unsigned int j=0; j<dum.size(); ++j) {
                  init_twice_active_var(dum[j], nlocal, localstateidx + j);
                }

                Matrix<MSScalar, 5, 1> du = Matrix<MSScalar, 5, 1>::Zero();
                for (unsigned int j=0; j<du.size(); ++j) {
                  init_twice_active_var(du[j], nlocal, localstateidx + 5 + j);
                }
                
                MSScalar dbeta(0.);
                init_twice_active_var(dbeta, nlocal, localparmidx);

                MSScalar dxi(0.);
                init_twice_active_var(dxi, nlocal, localparmidx + 1);
                            
                const Matrix<MSScalar, 5, 1> dprop = dx0.cast<MSScalar>() + Hpstate*du - Hmstate*Fstate*dum - Hmstate*Fb*dbeta - Hmstate*Fxi*dxi;
                const MSScalar chisq = dprop.transpose()*Qinv*dprop;
                
                chisq0val += chisq.value().value();
                  
                auto const& gradlocal = chisq.value().derivatives();
                //fill local hessian
                Matrix<double, nlocal, nlocal> hesslocal;
                for (unsigned int j=0; j<nlocal; ++j) {
                  hesslocal.row(j) = chisq.derivatives()[j].derivatives();
                }
                
                constexpr std::array<unsigned int, 2> localsizes = {{ nlocalstate, nlocalparms }};
                constexpr std::array<unsigned int, 2> localidxs = {{ localstateidx, localparmidx }};
                const std::array<unsigned int, 2> fullidxs = {{ fullstateidx, fullparmidx }};
                
                for (unsigned int iidx = 0; iidx < localidxs.size(); ++iidx) {
                  gradfull.segment(fullidxs[iidx], localsizes[iidx]) += gradlocal.segment(localidxs[iidx], localsizes[iidx]);
                  for (unsigned int jidx = 0; jidx < localidxs.size(); ++jidx) {
                    hessfull.block(fullidxs[iidx], fullidxs[jidx], localsizes[iidx], localsizes[jidx]) += hesslocal.block(localidxs[iidx], localidxs[jidx], localsizes[iidx], localsizes[jidx]);
                  }
                }
                
              }
              
              const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(6, parmdetid));
              globalidxv[parmidx] = bfieldglobalidx;
              parmidx++;
              
              const unsigned int elossglobalidx = detidparms.at(std::make_pair(7, parmdetid));
              globalidxv[parmidx] = elossglobalidx;
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
                
                const unsigned int fullstateidx = trackstateidx + 3 + 5*ihit + 3;
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
                
                chisq0val += chisq.value().value();

                auto const& gradlocal = chisq.value().derivatives();
                //fill local hessian
                Matrix<double, nlocal, nlocal> hesslocal;
                for (unsigned int j=0; j<nlocal; ++j) {
                  hesslocal.row(j) = chisq.derivatives()[j].derivatives();
                }
                
                constexpr std::array<unsigned int, 2> localsizes = {{ nlocalstate, nlocalparms }};
                constexpr std::array<unsigned int, 2> localidxs = {{ localstateidx, localparmidx }};
                const std::array<unsigned int, 2> fullidxs = {{ fullstateidx, fullparmidx }};
                
                for (unsigned int iidx = 0; iidx < localidxs.size(); ++iidx) {
                  gradfull.segment(fullidxs[iidx], localsizes[iidx]) += gradlocal.segment(localidxs[iidx], localsizes[iidx]);
                  for (unsigned int jidx = 0; jidx < localidxs.size(); ++jidx) {
                    hessfull.block(fullidxs[iidx], fullidxs[jidx], localsizes[iidx], localsizes[jidx]) += hesslocal.block(localidxs[iidx], localidxs[jidx], localsizes[iidx], localsizes[jidx]);
                  }
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
  //               gradfull.segment<nlocalstate>(fullstateidx) += gradlocal.head(nlocalstate);
  //               gradfull.segment<nlocalparms>(fullparmidx) += gradlocal.segment(localparmidx, nlocalparms);
  // 
  //               //fill global hessian (upper triangular blocks only)
  //               hessfull.block<nlocalstate,nlocalstate>(fullstateidx, fullstateidx) += hesslocal.topLeftCorner(nlocalstate,nlocalstate);
  //               hessfull.block<nlocalstate,nlocalparms>(fullstateidx, fullparmidx) += hesslocal.topRightCorner(nlocalstate, nlocalparms);
  //               hessfull.block<nlocalparms, nlocalparms>(fullparmidx, fullparmidx) += hesslocal.bottomRightCorner(nlocalparms, nlocalparms);
                
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
          
          trackstateidx += 3 + 5*tracknhits;
        }
        
        if (!valid) {
          break;
        }
        
  //       MatrixXd massjac;

        // add mass constraint to gbl fit
        if (doMassConstraint_) {
//         if (false) {
          constexpr unsigned int nvtxstate = 6;
          constexpr unsigned int nlocalstate = 3;
          
          constexpr unsigned int nlocal = nvtxstate + nlocalstate;
          
          constexpr unsigned int localvtxidx = 0;
          constexpr unsigned int localstateidx = localvtxidx + nvtxstate;
          
          constexpr unsigned int fullvtxidx = 0;
          const unsigned int fullstateidx = trackstateidxarr[1];
          
          using MScalar = AANT<double, nlocal>;
          
          //TODO optimize to avoid recomputation of FTS
  //         const FreeTrajectoryState refFts0 = outparts[0]->currentState().freeTrajectoryState();
  //         const FreeTrajectoryState refFts1 = outparts[1]->currentState().freeTrajectoryState();
          
          const FreeTrajectoryState &refFts0 = refftsarr[0];
          const FreeTrajectoryState &refFts1 = refftsarr[1];
          
          const ROOT::Math::PxPyPzMVector mom0(refFts0.momentum().x(),
                                                  refFts0.momentum().y(),
                                                  refFts0.momentum().z(),
                                                  mmu);
          
          const ROOT::Math::PxPyPzMVector mom1(refFts1.momentum().x(),
                                                  refFts1.momentum().y(),
                                                  refFts1.momentum().z(),
                                                  mmu);
          
          const double massval = (mom0 + mom1).mass();
          
//           const double mrval = 1./massval/massval;
          
//           const double mrconstraintval = 1./massconstraintval/massconstraintval;
          
//           const double 
          
          const Matrix<double, 5, 7> &FdFmref0 = FdFmrefarr[0];
          const Matrix<double, 5, 7> &FdFmref1 = FdFmrefarr[1];
          
          JacobianCurvilinearToCartesian curv2cartref0(refFts0.parameters());
          auto const &jacCurv2Cartref0 = Map<const Matrix<double, 6, 5, RowMajor>>(curv2cartref0.jacobian().Array());
          
          JacobianCurvilinearToCartesian curv2cartref1(refFts1.parameters());
          auto const &jacCurv2Cartref1 = Map<const Matrix<double, 6, 5, RowMajor>>(curv2cartref1.jacobian().Array());
          
//           const Matrix<double, 1, 6> m2jac = massJacobian(refFts0, refFts1, mmu);
          
//           std::cout << "massval = " << massval << std::endl;
          
          const Matrix<double, 1, 6> mjacalt = massJacobianAlt(refFts0, refFts1, mmu);
//           const Matrix<double, 1, 6> mjacalt = mrJacobian(refFts0, refFts1, mmu);
          
//           const Matrix<double, 1, 3> mjac0 = m2jac.leftCols<3>()*jacCurv2Cartref0.bottomLeftCorner<3, 3>();
//           const Matrix<double, 1, 3> mjac1 = m2jac.rightCols<3>()*jacCurv2Cartref1.bottomLeftCorner<3, 3>();
          
//           const Matrix<double, 1, 3> mjacalt0 = mjacalt.leftCols<3>();
//           const Matrix<double, 1, 3> mjacalt1 = mjacalt.rightCols<3>();
          
//           std::cout << "mjac0" << std::endl;
//           std::cout << mjac0 << std::endl;
//           std::cout << "mjacalt0" << std::endl;
//           std::cout << mjacalt0 << std::endl;
// 
//           std::cout << "mjac1" << std::endl;
//           std::cout << mjac1 << std::endl;
//           std::cout << "mjacalt1" << std::endl;
//           std::cout << mjacalt1 << std::endl;
                            
          // initialize active scalars for common vertex parameters
          Matrix<MScalar, 3, 1> dvtx = Matrix<MScalar, 3, 1>::Zero();
          for (unsigned int j=0; j<dvtx.size(); ++j) {
            init_twice_active_var(dvtx[j], nlocal, localvtxidx + j);
          }

          // initialize active scalars for state parameters
          // (first track is index together with vertex parameters)
          
          Matrix<MScalar, 3, 1> dmomcurv0 = Matrix<MScalar, 3, 1>::Zero();
          for (unsigned int j=0; j<dmomcurv0.size(); ++j) {
            init_twice_active_var(dmomcurv0[j], nlocal, localvtxidx + 3 + j);
          }
          
          Matrix<MScalar, 3, 1> dmomcurv1 = Matrix<MScalar, 3, 1>::Zero();
          for (unsigned int j=0; j<dmomcurv1.size(); ++j) {
            init_twice_active_var(dmomcurv1[j], nlocal, localstateidx + j);
          }
          
//           const Matrix<MScalar, 3, 1> dmom0 = jacCurv2Cartref0.bottomLeftCorner<3, 3>().cast<MScalar>()*dmomcurv0;
//           const Matrix<MScalar, 3, 1> dmom1 = jacCurv2Cartref1.bottomLeftCorner<3, 3>().cast<MScalar>()*dmomcurv1;
          
          // resonance width
  //         const MScalar invSigmaMsq(0.25/massConstraint_/massConstraint_/massConstraintWidth_/massConstraintWidth_);
  //         const MScalar dmsq0 = MScalar(m0*m0 - massConstraint_*massConstraint_);
          
          const MScalar invSigmaMsq(1./massConstraintWidth_/massConstraintWidth_);
          const MScalar dmsq0 = MScalar(massval - massconstraintval);
//           const MScalar invSigmaMsq(0.25*std::pow(massval, 6)/massConstraintWidth_/massConstraintWidth_);
//           const MScalar dmsq0 = MScalar(mrval - mrconstraintval);
//           const MScalar dmsq0 = MScalar(massval - massConstraint_);
//           const MScalar dmsq0 = MScalar(0.);
  //         const MScalar dmsq0 = MScalar(m0 - massConstraint_);
          
          
  //         std::cout << "invSigmaMsq = " << invSigmaMsq.value().value() << std::endl;
          
//           const MScalar dmsqtrack0 = (m2jac.leftCols<3>().cast<MScalar>()*dmom0)[0];
//           const MScalar dmsqtrack1 = (m2jac.rightCols<3>().cast<MScalar>()*dmom1)[0];
          
          const MScalar dmsqtrack0 = (mjacalt.leftCols<3>().cast<MScalar>()*dmomcurv0)[0];
          const MScalar dmsqtrack1 = (mjacalt.rightCols<3>().cast<MScalar>()*dmomcurv1)[0];
          
          const MScalar dmsq = dmsq0 + dmsqtrack0 + dmsqtrack1;
          
  //         const MScalar chisq = dmsq*invSigmaMsq*dmsq;
          const MScalar chisq = invSigmaMsq*dmsq*dmsq;
          
          chisq0val += chisq.value().value();
          
          auto const& gradlocal = chisq.value().derivatives();
          //fill local hessian
          Matrix<double, nlocal, nlocal> hesslocal;
          for (unsigned int j=0; j<nlocal; ++j) {
            hesslocal.row(j) = chisq.derivatives()[j].derivatives();
          }
          
          constexpr std::array<unsigned int, 2> localsizes = {{ nvtxstate, nlocalstate }};
          constexpr std::array<unsigned int, 2> localidxs = {{ localvtxidx, localstateidx }};
          const std::array<unsigned int, 2> fullidxs = {{ fullvtxidx, fullstateidx }};
          
          for (unsigned int iidx = 0; iidx < localidxs.size(); ++iidx) {
            gradfull.segment(fullidxs[iidx], localsizes[iidx]) += gradlocal.segment(localidxs[iidx], localsizes[iidx]);
            for (unsigned int jidx = 0; jidx < localidxs.size(); ++jidx) {
              hessfull.block(fullidxs[iidx], fullidxs[jidx], localsizes[iidx], localsizes[jidx]) += hesslocal.block(localidxs[iidx], localidxs[jidx], localsizes[iidx], localsizes[jidx]);
            }
          }
          
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
        
        assert(trackstateidx == nstateparms);
        assert(parmidx == (nparsBfield + nparsEloss));
        assert(alignmentparmidx == nparsAlignment);
        
  //       if (nhits != nvalid) {
  //         continue;
  //       }

        auto freezeparm = [&](unsigned int idx) {
          gradfull[idx] = 0.;
          hessfull.row(idx) *= 0.;
          hessfull.col(idx) *= 0.;
          hessfull(idx,idx) = 1e6;
        };
        
        if (fitFromGenParms_) {
          for (unsigned int i=0; i<3; ++i) {
            freezeparm(i);
          }
          for (unsigned int id = 0; id < 2; ++id) {
            for (unsigned int i=0; i<3; ++i) {
              freezeparm(trackstateidxarr[id] + i);
            }
          }
        }
        
//         if (fitFromGenParms_) {
//           for (unsigned int id = 0; id < 2; ++id) {
//             for (unsigned int i=1; i<3; ++i) {
//               freezeparm(trackstateidxarr[id] + i);
//             }
//           }
//         }
        
        //now do the expensive calculations and fill outputs
        
        //symmetrize the matrix (previous block operations do not guarantee that the needed blocks are filled)
        //TODO handle this more efficiently?
  //       hessfull.triangularView<StrictlyLower>() = hessfull.triangularView<StrictlyUpper>().transpose();
        
  //       for (unsigned int i=0; i<3; ++i) {
  //         gradfull[i] = 0.;
  //         hessfull.row(i) *= 0.;
  //         hessfull.col(i) *= 0.;
  //         hessfull(i,i) = 1e6;
  //       }
        
  //       for (auto trackstateidx : trackstateidxarr) {
  //         for (unsigned int i = trackstateidx; i < (trackstateidx + 1); ++i) {
  //           gradfull[i] = 0.;
  //           hessfull.row(i) *= 0.;
  //           hessfull.col(i) *= 0.;
  //           hessfull(i,i) = 1e6;
  //         }
  //       }
        
  //       {
  //         unsigned int i = trackstateidxarr[1];
  //         gradfull[i] = 0.;
  //         hessfull.row(i) *= 0.;
  //         hessfull.col(i) *= 0.;
  //         hessfull(i,i) = 1e6; 
  //       }
  //       
        
  //       std::cout << "gradfull:" << std::endl;
  //       std::cout << gradfull << std::endl;
  //       
  //       std::cout << "gradfull.head(nstateparms):" << std::endl;
  //       std::cout << gradfull.head(nstateparms) << std::endl;
  // 
  //       std::cout << "gradfull.tail(npars):" << std::endl;
  //       std::cout << gradfull.tail(npars) << std::endl;
  //       
  //       std::cout << "hessfull.diagonal():" << std::endl;
  //       std::cout << hessfull.diagonal() << std::endl;
        
        auto const& dchisqdx = gradfull.head(nstateparms);
        auto const& dchisqdparms = gradfull.tail(npars);
        
        auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
        auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
        auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
        

        
        Cinvd.compute(d2chisqdx2);
        
        dxfull = -Cinvd.solve(dchisqdx);
        dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
        
//         dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
        
    //     if (debugprintout_) {
    //       std::cout << "dxrefdparms" << std::endl;
    //       std::cout << dxdparms.leftCols<5>() << std::endl;
    //     }
        
//         grad = dchisqdparms + dxdparms*dchisqdx;
        //TODO check the simplification
    //     hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
//         hess = d2chisqdparms2 + dxdparms*d2chisqdxdparms;
        
        const Matrix<double, 1, 1> deltachisq = dchisqdx.transpose()*dxfull + 0.5*dxfull.transpose()*d2chisqdx2*dxfull;
        
//         std::cout << "iiter = " << iiter << ", deltachisq = " << deltachisq[0] << std::endl;
//         
//         SelfAdjointEigenSolver<MatrixXd> es(d2chisqdx2, EigenvaluesOnly);
//         const double condition = es.eigenvalues()[nstateparms-1]/es.eigenvalues()[0];
//         std::cout << "eigenvalues:" << std::endl;
//         std::cout << es.eigenvalues().transpose() << std::endl;
//         std::cout << "condition: " << condition << std::endl;
        
        chisqval = chisq0val + deltachisq[0];
        
        ndof = 5*nhits + nvalid + nvalidalign2d - nstateparms;
        
        if (bsConstraint_) {
          ndof += 3;
        }
        
        if (doMassConstraint_) {
          ++ndof;
        }
        
  //       std::cout << "dchisqdparms.head<6>()" << std::endl;
  //       std::cout << dchisqdparms.head<6>() << std::endl;
  //       
  //       std::cout << "grad.head<6>()" << std::endl;
  //       std::cout << grad.head<6>() << std::endl;
  //       
  //       std::cout << "d2chisqdparms2.topLeftCorner<6, 6>():" << std::endl;
  //       std::cout << d2chisqdparms2.topLeftCorner<6, 6>() << std::endl;
  //       std::cout << "hess.topLeftCorner<6, 6>():" << std::endl;
  //       std::cout << hess.topLeftCorner<6, 6>() << std::endl;
  //       
  //       std::cout << "dchisqdparms.segment<6>(nparsBfield+nparsEloss)" << std::endl;
  //       std::cout << dchisqdparms.segment<6>(nparsBfield+nparsEloss) << std::endl;
  //       
  //       std::cout << "grad.segment<6>(nparsBfield+nparsEloss)" << std::endl;
  //       std::cout << grad.segment<6>(nparsBfield+nparsEloss) << std::endl;
  //       
  //       std::cout << "d2chisqdparms2.block<6, 6>(nparsBfield+nparsEloss, nparsBfield+nparsEloss):" << std::endl;
  //       std::cout << d2chisqdparms2.block<6, 6>(nparsBfield+nparsEloss, nparsBfield+nparsEloss) << std::endl;
  //       std::cout << "hess.block<6, 6>(nparsBfield+nparsEloss, nparsBfield+nparsEloss):" << std::endl;
  //       std::cout << hess.block<6, 6>(nparsBfield+nparsEloss, nparsBfield+nparsEloss) << std::endl;
  // //       
  //       
  //       std::cout << "d2chisqdparms2.block<6, 6>(trackparmidxarr[1], trackparmidxarr[1]):" << std::endl;
  //       std::cout << d2chisqdparms2.block<6, 6>(trackparmidxarr[1], trackparmidxarr[1]) << std::endl;
  //       std::cout << "hess.block<6, 6>(trackparmidxarr[1], trackparmidxarr[1]):" << std::endl;
  //       std::cout << hess.block<6, 6>(trackparmidxarr[1], trackparmidxarr[1]) << std::endl;
  //       
  //       std::cout << "d2chisqdparms2.bottomRightCorner<6, 6>():" << std::endl;
  //       std::cout << d2chisqdparms2.bottomRightCorner<6, 6>() << std::endl;
  //       std::cout << "hess.bottomRightCorner<6, 6>():" << std::endl;
  //       std::cout << hess.bottomRightCorner<6, 6>() << std::endl;

  //       const double 
  // //       const double corxi0plusminus = hess(1, trackparmidxarr[1] + 1)/std::sqrt(hess(1,1)*hess(trackparmidxarr[1] + 1, trackparmidxarr[1] + 1));
  // //       const double corxi1plusminus = hess(3, trackparmidxarr[1] + 3)/std::sqrt(hess(3,3)*hess(trackparmidxarr[1] + 3, trackparmidxarr[1] + 3));
  //       
  //       const double cor01plus = hess(1, 3)/std::sqrt(hess(1, 1)*hess(3, 3));
  // //       const double cor01minus = hess(trackparmidxarr[1] + 1, trackparmidxarr[1] + 3)/std::sqrt(hess(trackparmidxarr[1] + 1, trackparmidxarr[1] + 1)*hess(trackparmidxarr[1] + 3, trackparmidxarr[1] + 3));
  // 
  //       const double cor12plus = hess(3, 5)/std::sqrt(hess(3, 3)*hess(5, 5));
  // //       const double cor12minus = hess(trackparmidxarr[1] + 3, trackparmidxarr[1] + 5)/std::sqrt(hess(trackparmidxarr[1] + 3, trackparmidxarr[1] + 3)*hess(trackparmidxarr[1] + 5, trackparmidxarr[1] + 5));
  //       
  // //       std::cout << "corxi0plusminus = " << corxi0plusminus << std::endl;
  // //       std::cout << "corxi1plusminus = " << corxi1plusminus << std::endl;
  //       std::cout << "cor01plus = " << cor01plus << std::endl;
  // //       std::cout << "cor01minus = " << cor01minus << std::endl;
  //       std::cout << "cor12plus = " << cor12plus << std::endl;
  // //       std::cout << "cor12minus = " << cor12minus << std::endl;
        
  //       std::cout << "hess(1, 1)" << std::endl;
  //       std::cout << hess(1, 1) << std::endl;
  //       std::cout << "hess(trackparmidxarr[1] + 1, trackparmidxarr[1] + 1)" << std::endl;
  //       std::cout << hess(trackparmidxarr[1] + 1, trackparmidxarr[1] + 1) << std::endl;
  //       std::cout << "hess(1, trackparmidxarr[1] + 1)" << std::endl;
  //       std::cout << hess(1, trackparmidxarr[1] + 1) << std::endl;
        
        // compute final kinematics
        
        kinTree->movePointerToTheTop();
        RefCountedKinematicVertex dimu_vertex = kinTree->currentDecayVertex();
        
        Jpsikin_x = dimu_vertex->position().x();
        Jpsikin_y = dimu_vertex->position().y();
        Jpsikin_z = dimu_vertex->position().z();
        
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
        std::array<Vector3d, 2> mucurvarr;
        std::array<int, 2> muchargearr;
        
  //       std::cout << dimu_vertex->position() << std::endl;
        
        // apply the GBL fit results to the muon kinematics
        for (unsigned int id = 0; id < 2; ++id) {
  //         auto const &refFts = outparts[id]->currentState().freeTrajectoryState();
          auto const &refFts = refftsarr[id];
//           auto const &jac = jacarr[id];
          unsigned int trackstateidx = trackstateidxarr[id];
          
          JacobianCurvilinearToCartesian curv2cart(refFts.parameters());
          const AlgebraicMatrix65& jac = curv2cart.jacobian();
          const AlgebraicVector6 glob = refFts.parameters().vector();
          
          const Matrix<double, 3, 1> posupd = Map<const Matrix<double, 6, 1>>(glob.Array()).head<3>() + dxfull.head<3>();
          
          const Matrix<double, 3, 1> momupd = Map<const Matrix<double, 6, 1>>(glob.Array()).tail<3>() + Map<const Matrix<double, 6, 5, RowMajor>>(jac.Array()).bottomLeftCorner<3, 3>()*dxfull.segment<3>(trackstateidx);
          
          const GlobalPoint pos(posupd[0], posupd[1], posupd[2]);
          const GlobalVector mom(momupd[0], momupd[1], momupd[2]);
          const double charge = std::copysign(1., refFts.charge()/refFts.momentum().mag() + dxfull[trackstateidx]);
    //         std::cout << "before update: reffts:" << std::endl;
    //         std::cout << refFts.parameters().vector() << std::endl;
    //         std::cout << "charge " << refFts.charge() << std::endl;
//           updFts = FreeTrajectoryState(pos, mom, charge, field);

          
          muarr[id] = ROOT::Math::PxPyPzMVector(momupd[0], momupd[1], momupd[2], mmu);
          muchargearr[id] = charge;
                  
          auto &refParms = mucurvarr[id];
//           CurvilinearTrajectoryParameters curvparms(refFts.position(), refFts.momentum(), refFts.charge());
          CurvilinearTrajectoryParameters curvparms(pos, mom, charge);
//           refParms << curvparms.Qbp(), curvparms.lambda(), curvparms.phi(), curvparms.xT(), curvparms.yT();
          refParms << curvparms.Qbp(), curvparms.lambda(), curvparms.phi();
//           refParms += dxcurv;

        }
        
        // *TODO* better handling of this case?
        if ( (muchargearr[0] + muchargearr[1]) != 0) {
          valid = false;
          break;
        }

        const unsigned int idxplus = muchargearr[0] > 0 ? 0 : 1;
        const unsigned int idxminus = muchargearr[0] > 0 ? 1 : 0;
        
        Muplus_pt = muarr[idxplus].pt();
        Muplus_eta = muarr[idxplus].eta();
        Muplus_phi = muarr[idxplus].phi();
        
        Muminus_pt = muarr[idxminus].pt();
        Muminus_eta = muarr[idxminus].eta();
        Muminus_phi = muarr[idxminus].phi();
        
        Mupluskin_pt = outparts[idxplus]->currentState().globalMomentum().perp();
        Mupluskin_eta = outparts[idxplus]->currentState().globalMomentum().eta();
        Mupluskin_phi = outparts[idxplus]->currentState().globalMomentum().phi();
        
        Muminuskin_pt = outparts[idxminus]->currentState().globalMomentum().perp();
        Muminuskin_eta = outparts[idxminus]->currentState().globalMomentum().eta();
        Muminuskin_phi = outparts[idxminus]->currentState().globalMomentum().phi();
        
  //       std::cout << "Muplus pt, eta, phi = " << Muplus_pt << ", " << Muplus_eta << ", " << Muplus_phi << std::endl;
  //       std::cout << "Muminus pt, eta, phi = " << Muminus_pt << ", " << Muminus_eta << ", " << Muminus_phi << std::endl;
        
        Map<Matrix<float, 3, 1>>(Muplus_refParms.data()) = mucurvarr[idxplus].cast<float>();
        Map<Matrix<float, 3, 1>>(MuMinus_refParms.data()) = mucurvarr[idxminus].cast<float>();
        
//         std::cout << "nstateparms = " << nstateparms << std::endl;
//         std::cout << "dxdparms " << dxdparms.rows() << " " << dxdparms.cols() << std::endl;
        
        Muplus_jacRef.resize(3*npars);
        Map<Matrix<float, 3, Dynamic, RowMajor>>(Muplus_jacRef.data(), 3, npars) = dxdparms.block(0, trackstateidxarr[idxplus], npars, 3).transpose().cast<float>();
        
        Muminus_jacRef.resize(3*npars);
        Map<Matrix<float, 3, Dynamic, RowMajor>>(Muminus_jacRef.data(), 3, npars) = dxdparms.block(0, trackstateidxarr[idxminus], npars, 3).transpose().cast<float>();
//         
//         (jacarr[idxplus].topLeftCorner(5, nstateparms)*dxdparms.transpose() + jacarr[idxplus].topRightCorner(5, npars)).cast<float>();
//         
//         Muminus_jacRef.resize(3*npars);
//         Map<Matrix<float, 3, Dynamic, RowMajor>>(Muminus_jacRef.data(), 3, npars) = (jacarr[idxminus].topLeftCorner(5, nstateparms)*dxdparms.transpose() + jacarr[idxminus].topRightCorner(5, npars)).cast<float>();
        
        //TODO fix this
//         Muplus_jacRef.resize(5*npars);
//         Map<Matrix<float, 5, Dynamic, RowMajor>>(Muplus_jacRef.data(), 5, npars) = (jacarr[idxplus].topLeftCorner(5, nstateparms)*dxdparms.transpose() + jacarr[idxplus].topRightCorner(5, npars)).cast<float>();
//         
//         Muminus_jacRef.resize(5*npars);
//         Map<Matrix<float, 5, Dynamic, RowMajor>>(Muminus_jacRef.data(), 5, npars) = (jacarr[idxminus].topLeftCorner(5, nstateparms)*dxdparms.transpose() + jacarr[idxminus].topRightCorner(5, npars)).cast<float>();
        
        auto const jpsimom = muarr[0] + muarr[1];
        
        Jpsi_pt = jpsimom.pt();
        Jpsi_eta = jpsimom.eta();
        Jpsi_phi = jpsimom.phi();
        Jpsi_mass = jpsimom.mass();
        
        Muplus_nhits = nhitsarr[idxplus];
        Muplus_nvalid = nvalidarr[idxplus];
        Muplus_nvalidpixel = nvalidpixelarr[idxplus];
        
        Muminus_nhits = nhitsarr[idxminus];
        Muminus_nvalid = nvalidarr[idxminus];
        Muminus_nvalidpixel = nvalidpixelarr[idxminus];
        
        const ROOT::Math::PxPyPzMVector mompluskin(outparts[idxplus]->currentState().globalMomentum().x(),
                                                          outparts[idxplus]->currentState().globalMomentum().y(),
                                                          outparts[idxplus]->currentState().globalMomentum().z(),
                                                          mmu);
        
        const ROOT::Math::PxPyPzMVector momminuskin(outparts[idxminus]->currentState().globalMomentum().x(),
                                                          outparts[idxminus]->currentState().globalMomentum().y(),
                                                          outparts[idxminus]->currentState().globalMomentum().z(),
                                                          mmu);
        
        auto const jpsimomkin = mompluskin + momminuskin;
        
        Jpsikin_pt = jpsimomkin.pt();
        Jpsikin_eta = jpsimomkin.eta();
        Jpsikin_phi = jpsimomkin.phi();
        Jpsikin_mass = jpsimomkin.mass();
        
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


        niter = iiter + 1;
        edmval = -deltachisq[0];
        
//         std::cout << "iiter = " << iiter << " edmval = " << edmval << std::endl;
        
        if (iiter > 1 && std::abs(deltachisq[0])<1e-3) {
          break;
        }
    
      }
      
      if (!valid) {
        continue;
      }
    
      auto const& dchisqdx = gradfull.head(nstateparms);
      auto const& dchisqdparms = gradfull.tail(npars);
      
      auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
      auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
      auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
      
      
  //     if (debugprintout_) {
  //       std::cout << "dxrefdparms" << std::endl;
  //       std::cout << dxdparms.leftCols<5>() << std::endl;
  //     }
      
      grad = dchisqdparms + dxdparms*dchisqdx;
      //TODO check the simplification
  //     hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
      hess = d2chisqdparms2 + dxdparms*d2chisqdxdparms;
  
//       for (unsigned int iparm = 0; iparm < npars; ++iparm) {
//         if (detidparmsrev[globalidxv[iparm]].first != 7) {
//           hess.row(iparm) *= 0.;
//           hess.col(iparm) *= 0.;
//           hess(iparm, iparm) = 1e6;
//         }
//       }
      
//       SelfAdjointEigenSolver<MatrixXd> es(hess, EigenvaluesOnly);
//       const double condition = es.eigenvalues()[nstateparms-1]/es.eigenvalues()[0];
//       std::cout << "hess eigenvalues:" << std::endl;
//       std::cout << es.eigenvalues().transpose() << std::endl;
//       std::cout << "condition: " << condition << std::endl;
      
//       std::cout << "hess diagonal:" << std::endl;
//       std::cout << hess.diagonal().transpose() << std::endl;
//       
//       assert(es.eigenvalues()[0] > -1e-5);
//       assert(hess.diagonal().minCoeff() > 0.);
      
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
      
      
      assert(globalidxv.size() == (2*Muplus_nhits + 2*Muminus_nhits + 2*Muplus_nvalid + 2*Muminus_nvalid + Muplus_nvalidpixel + Muminus_nvalidpixel));

//       hessv.resize(npars*npars);
//       Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(hessv.data(), npars, npars) = hess.cast<float>();
      
      if (fillTrackTree_) {
        tree->Fill();
      }
      

      

      
//       const Matrix3d covvtx = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<3,3>();
//       
//       const double covqop0 = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))(trackstateidxarr[0], trackstateidxarr[0]);
//       const double covqop1 = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))(trackstateidxarr[1], trackstateidxarr[1]);
//       
//       const double covqop0kin = outparts[0]->currentState().freeTrajectoryState().curvilinearError().matrix()(0,0);
//       const double covqop1kin = outparts[1]->currentState().freeTrajectoryState().curvilinearError().matrix()(0,0);
//       
// //       Matrix<double, 1, 1> covmass = 2.*massjac*Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))*massjac.transpose();
//       
// //       const VectorXd cinvrow0 = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).row(0).head(nstateparms);
// //       
//       std::cout << "kinfit covariance:" << std::endl;
//       std::cout << dimu_vertex->error().matrix() << std::endl;
//       
//       std::cout << "GBL covariance:" << std::endl;
//       std::cout << 2.*covvtx << std::endl;
//       
//       std::cout << "kinfit qop0 covariance:" << std::endl;
//       std::cout << covqop0kin << std::endl;
//       
//       std::cout << "GBL qop0 covariance:" << std::endl;
//       std::cout << 2.*covqop0 << std::endl;
//       
//       std::cout << "kinfit qop1 covariance:" << std::endl;
//       std::cout << covqop1kin << std::endl;
//       
//       std::cout << "GBL qop1 covariance:" << std::endl;
//       std::cout << 2.*covqop1 << std::endl;
      
//       std::cout << "dqop0 beamline" << std::endl;
//       std::cout << dxfull[trackstateidxarr[0]] << std::endl;
//       std::cout << "dqop0 first layer" << std::endl;
//       std::cout << dxfull[trackstateidxarr[0]+3] << std::endl;
//       std::cout << "dqop0 second layer" << std::endl;
//       std::cout << dxfull[trackstateidxarr[0]+6] << std::endl;
//       
//       std::cout << "dqop1 beamline" << std::endl;
//       std::cout << dxfull[trackstateidxarr[1]] << std::endl;
//       std::cout << "dqop1 first layer" << std::endl;
//       std::cout << dxfull[trackstateidxarr[1]+3] << std::endl;
//       std::cout << "dqop1 second layer" << std::endl;
//       std::cout << dxfull[trackstateidxarr[1]+6] << std::endl;
//       
//       std::cout << "sigmam" << std::endl;
//       std::cout << std::sqrt(covmass[0]) << std::endl;
      
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

Matrix<double, 1, 6> ResidualGlobalCorrectionMakerTwoTrackG4e::massJacobian(const FreeTrajectoryState &state0, const FreeTrajectoryState &state1, double dmass) const {
  Matrix<double, 1, 6> res = Matrix<double, 6, 1>::Zero();

  const double e0 = std::sqrt(state0.momentum().mag2() + dmass*dmass);
  const double e1 = std::sqrt(state1.momentum().mag2() + dmass*dmass);
  
  // dm^2/dp0x
  res(0, 0) = 2.*e1*state0.momentum().x()/e0 - 2.*state1.momentum().x();
  // dm^2/dp0y
  res(0, 1) = 2.*e1*state0.momentum().y()/e0 - 2.*state1.momentum().y();
  // dm^2/dp0z
  res(0, 2) = 2.*e1*state0.momentum().z()/e0 - 2.*state1.momentum().z();
  
  // d^m/dp1x
  res(0, 3) = 2.*e0*state1.momentum().x()/e1 - 2.*state0.momentum().x();
  // d^m/dp1y
  res(0, 4) = 2.*e0*state1.momentum().y()/e1 - 2.*state0.momentum().y();
  // d^m/dp1z
  res(0, 5) = 2.*e0*state1.momentum().z()/e1 - 2.*state0.momentum().z();
  
  const double m = std::sqrt(2.*dmass*dmass + 2.*e0*e1 - 2.*state0.momentum().x()*state1.momentum().x() - 2.*state0.momentum().y()*state1.momentum().y() - 2.*state0.momentum().z()*state1.momentum().z());
  
  res *= 0.5/m;
  
  return res;
}


DEFINE_FWK_MODULE(ResidualGlobalCorrectionMakerTwoTrackG4e);
