// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/TkRadialStripTopology.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"
#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"



// #include "../interface/OffsetMagneticField.h"
// #include "../interface/ParmInfo.h"


#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "functions.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

using namespace Eigen;

constexpr unsigned int max_n = 25; //!< In order to avoid use of dynamic memory

//too big for stack :(
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 5*max_n> GlobalParameterMatrix;
// typedef Matrix<double, Dynamic, 1, 0, 5*max_n, 1> GlobalParameterVector;
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 2*max_n> AlignmentJacobianMatrix;
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 2*max_n> TransportJacobianMatrix;

typedef Matrix<double, 5, 5> Matrix5d;
typedef Matrix<double, 5, 1> Vector5d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<float, 5, 5> Matrix5f;
typedef Matrix<float, 5, 1> Vector5f;
typedef Matrix<unsigned int, Dynamic, 1> VectorXu;

typedef MatrixXd GlobalParameterMatrix;
typedef VectorXd GlobalParameterVector;
typedef MatrixXd AlignmentJacobianMatrix;
typedef MatrixXd TransportJacobianMatrix;
typedef MatrixXd ELossJacobianMatrix;


// struct ParmInfo {
//   int parmtype;
//   int subdet;
//   int layer;
//   float x;
//   float y;
//   float z;
//   float eta;
//   float phi;
//   float rho;
// };

//
// class declaration
//

class ResidualGlobalCorrectionMaker : public edm::EDAnalyzer
{
public:
  explicit ResidualGlobalCorrectionMaker(const edm::ParameterSet &);
  ~ResidualGlobalCorrectionMaker();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  template <unsigned int D>
  AlgebraicVector5 lupdate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  AlgebraicVector5 update(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> inputTrack_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrackOrig_;
  edm::EDGetTokenT<std::vector<int> > inputIndices_;
  edm::EDGetTokenT<reco::BeamSpot> inputBs_;

  TFile *fout;
  TTree *tree;

  float trackEta;
  float trackPhi;
  float trackPt;
  float trackPtErr;
  float trackCharge;

  float genPt;
  float genEta;
  float genPhi;
  float genCharge;
  
  unsigned int nParms;
  unsigned int nJacRef;
  unsigned int nSym;
  
  std::array<float, 5> trackParms;
  std::array<float, 25> trackCov;
  
  std::array<float, 5> refParms;
  std::array<float, 25> refCov;
  
  std::array<float, 5> genParms;
  
  std::vector<float> gradv;
  std::vector<float> jacrefv;
  std::vector<unsigned int> globalidxv;
  
  std::vector<float> hesspackedv;
  
  std::map<std::pair<int, DetId>, unsigned int> detidparms;
  
  unsigned int run;
  unsigned int lumi;
  unsigned long long event;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ResidualGlobalCorrectionMaker::ResidualGlobalCorrectionMaker(const edm::ParameterSet &iConfig)

{
  //now do what ever initialization is needed
  inputTraj_ = consumes<std::vector<Trajectory>>(edm::InputTag("TrackRefitter"));
  GenParticlesToken_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));
  inputTrack_ = consumes<TrajTrackAssociationCollection>(edm::InputTag("TrackRefitter"));
  inputIndices_ = consumes<std::vector<int> >(edm::InputTag("TrackRefitter"));
  inputTrackOrig_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  inputBs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));


  fout = new TFile("trackTreeGrads.root", "RECREATE");
//   fout = new TFile("trackTreeGradsdebug.root", "RECREATE");
//   fout = new TFile("trackTreeGradsdebug2.root", "RECREATE");
  //TODO this needs a newer root version
//   fout->SetCompressionAlgorithm(ROOT::kLZ4);
//   fout->SetCompressionLevel(3);
  tree = new TTree("tree", "tree");
  
  
  const int basketSize = 4*1024*1024;
  tree->SetAutoFlush(0);
  
  tree->Branch("trackPt", &trackPt, basketSize);
  tree->Branch("trackPtErr", &trackPtErr, basketSize);
  tree->Branch("trackEta", &trackEta, basketSize);
  tree->Branch("trackPhi", &trackPhi, basketSize);
  tree->Branch("trackCharge", &trackCharge, basketSize);
  //workaround for older ROOT version inability to store std::array automatically
  tree->Branch("trackParms", trackParms.data(), "trackParms[5]/F", basketSize);
  tree->Branch("trackCov", trackCov.data(), "trackCov[25]/F", basketSize);
  tree->Branch("refParms", refParms.data(), "refParms[5]/F", basketSize);
  tree->Branch("refCov", refCov.data(), "refCov[25]/F", basketSize);
  tree->Branch("genParms", genParms.data(), "genParms[5]/F", basketSize);

  tree->Branch("genPt", &genPt, basketSize);
  tree->Branch("genEta", &genEta, basketSize);
  tree->Branch("genPhi", &genPhi, basketSize);
  tree->Branch("genCharge", &genCharge, basketSize);
  
  tree->Branch("nParms", &nParms, basketSize);
  tree->Branch("nJacRef", &nJacRef, basketSize);
  
  tree->Branch("gradv", gradv.data(), "gradv[nParms]/F", basketSize);
  tree->Branch("globalidxv", globalidxv.data(), "globalidxv[nParms]/i", basketSize);
  tree->Branch("jacrefv",jacrefv.data(),"jacrefv[nJacRef]/F", basketSize);
  
  tree->Branch("nSym", &nSym, basketSize);
  
  tree->Branch("hesspackedv", hesspackedv.data(), "hesspackedv[nSym]/F", basketSize);
  
  tree->Branch("run", &run);
  tree->Branch("lumi", &lumi);
  tree->Branch("event", &event);

}

ResidualGlobalCorrectionMaker::~ResidualGlobalCorrectionMaker()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void ResidualGlobalCorrectionMaker::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  using namespace edm;

  Handle<std::vector<reco::GenParticle>> genPartCollection;
  iEvent.getByToken(GenParticlesToken_, genPartCollection);

  auto genParticles = *genPartCollection.product();

  // loop over gen particles

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  auto field = magfield.product();
  
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrh;
  iSetup.get<TransientRecHitRecord>().get("WithAngleAndTemplate",ttrh);
  
  Handle<TrajTrackAssociationCollection> trackH;
  iEvent.getByToken(inputTrack_, trackH);
  
  Handle<reco::TrackCollection> trackOrigH;
  iEvent.getByToken(inputTrackOrig_, trackOrigH);
  
  Handle<std::vector<int> > indicesH;
  iEvent.getByToken(inputIndices_, indicesH);
  
  Handle<std::vector<Trajectory> > trajH;
  iEvent.getByToken(inputTraj_, trajH);
  
  Handle<reco::BeamSpot> bsH;
  iEvent.getByToken(inputBs_, bsH);
  
  const reco::BeamSpot& bs = *bsH;
  
  const float mass = 0.105;
  const float maxDPhi = 1.6;
  PropagatorWithMaterial rPropagator(oppositeToMomentum, mass, field, maxDPhi, true, -1., false);
  PropagatorWithMaterial fPropagator(alongMomentum, mass, field, maxDPhi, true, -1., false);
  
//   KFUpdator updator;
//   TkClonerImpl hitCloner;
//   TKCloner const* cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->cloner()
//   TkClonerImpl const& cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(ttrh.product())->cloner();
//   TrajectoryStateCombiner combiner;
  
  run = iEvent.run();
  lumi = iEvent.luminosityBlock();
  event = iEvent.id().event();
  
  for (unsigned int j=0; j<trajH->size(); ++j) {
    const Trajectory& traj = (*trajH)[j];
    
    const edm::Ref<std::vector<Trajectory> > trajref(trajH, j);
    const reco::Track& track = *(*trackH)[trajref];
//     const reco::Track& trackOrig = (*trackOrigH)[(*indicesH)[j]];

    if (traj.isLooper()) {
      continue;
    }
    trackPt = track.pt();
    trackEta = track.eta();
    trackPhi = track.phi();
    trackCharge = track.charge();
    trackPtErr = track.ptError();
    
//     std::cout << "track pt: " << trackPt << " track eta: " << trackEta << " trackCharge: " << trackCharge << " qop: " << track.parameters()[0] << std::endl;
    
    auto const& tkparms = track.parameters();
    auto const& tkcov = track.covariance();
    trackParms.fill(0.);
    trackCov.fill(0.);
    //use eigen to fill raw memory
    Map<Vector5f>(trackParms.data()) = Map<const Vector5d>(tkparms.Array()).cast<float>();
    Map<Matrix<float, 5, 5, RowMajor> >(trackCov.data()).triangularView<Upper>() = Map<const Matrix<double, 5, 5, RowMajor> >(tkcov.Array()).cast<float>().triangularView<Upper>();
    
    const std::vector<TrajectoryMeasurement> &tms = traj.measurements();
    
    genPt = -99.;
    genEta = -99.;
    genPhi = -99.;
    genCharge = -99;
    genParms.fill(0.);
    for (std::vector<reco::GenParticle>::const_iterator g = genParticles.begin(); g != genParticles.end(); ++g)
    {

      float dR = deltaR(g->phi(), trackPhi, g->eta(), trackEta);

      if (dR < 0.15)
      {
        genPt = g->pt();
        genEta = g->eta();
        genPhi = g->phi();
        genCharge = g->charge();
        
        auto const& vtx = g->vertex();
        auto const& myBeamSpot = bs.position(vtx.z());
        
        //q/|p|
        genParms[0] = g->charge()/g->p();
        //lambda
        genParms[1] = M_PI_2 - g->momentum().theta();
        //phi
        genParms[2] = g->phi();
        //dxy
        genParms[3] = (-(vtx.x() - myBeamSpot.x()) * g->py() + (vtx.y() - myBeamSpot.y()) * g->px()) / g->pt();
        //dsz
        genParms[4] = (vtx.z() - myBeamSpot.z()) * g->pt() / g->p() -
           ((vtx.x() - myBeamSpot.x()) * g->px() + (vtx.y() - myBeamSpot.y()) * g->py()) / g->pt() * g->pz() / g->p();
      }
      else {
        continue;
      }
    }
    

    PropagationDirection rpropdir = traj.direction();
    PropagationDirection fpropdir = rpropdir == alongMomentum ? oppositeToMomentum : alongMomentum;
    
    //TODO properly handle the outside-in case
    assert(fpropdir == alongMomentum);
    
    const unsigned int nhits = tms.size();
    unsigned int npixhits = 0;

    //count pixel hits since this is needed to size the arrays
    for (unsigned int i = 0; i < tms.size(); ++i)
    {
      TrajectoryMeasurement const& tm = tms[i];
      auto const& hit = tm.recHit();
      const GeomDet *detectorG = globalGeometry->idToDet(hit->geographicalId());
      if (GeomDetEnumerators::isTrackerPixel(detectorG->subDetector())) {
        npixhits += 1;
      }
    }
    
    const unsigned int nstriphits = nhits-npixhits;
    const unsigned int nparsAlignment = nstriphits + 2*npixhits;
    const unsigned int nparsBfield = nhits - 1;
    const unsigned int nparsEloss = nhits - 1;
    const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
    
    const unsigned int nstateparms = 5*nhits;
    const unsigned int npropparms = 5*(nhits-1);
    const unsigned int nhitparms = 2*nhits;
    const unsigned int nmomparms = 3*(nhits-1);
    const unsigned int nposparms = 2*(nhits-1);
    constexpr unsigned int nrefparms = 5;
    
    //curvilinear to local jacobian (at two points), split into different pieces of the chi^2
    //to keep the size of the matrices to a minimum by projecting out only the relevant
    //components
    GlobalParameterMatrix Hh = GlobalParameterMatrix::Zero(nhitparms, nstateparms);
    GlobalParameterMatrix Hmom = GlobalParameterMatrix::Zero(nmomparms, nstateparms);
    GlobalParameterMatrix Hpos = GlobalParameterMatrix::Zero(nposparms, nstateparms);
    
    GlobalParameterMatrix Hpropmom = GlobalParameterMatrix::Zero(nmomparms, npropparms);
    GlobalParameterMatrix Hproppos = GlobalParameterMatrix::Zero(nposparms, npropparms);
    //alignment jacobian
    AlignmentJacobianMatrix A = AlignmentJacobianMatrix::Zero(nhitparms, nparsAlignment);
    //propagation jacobian wrt parameters
    GlobalParameterMatrix F = GlobalParameterMatrix::Zero(npropparms, nstateparms);
    //propagation jacobian back to track reference point wrt parameters
    Matrix<double, nrefparms, Dynamic> Fref = Matrix<double, nrefparms, Dynamic>::Zero(nrefparms, nstateparms);
    //energy loss jacobian wrt parameters
    GlobalParameterMatrix E = GlobalParameterMatrix::Zero(nmomparms, nmomparms);
    //propagation jacobian with respect to b-field
    TransportJacobianMatrix dF = TransportJacobianMatrix::Zero(npropparms, nparsBfield);
    //energy loss jacobian wrt energy loss
    ELossJacobianMatrix dE = ELossJacobianMatrix::Zero(nmomparms, nparsEloss);
//     //identity matrix
//     GlobalParameterMatrix const I = GlobalParameterMatrix::Identity(5*nhits, 5*nhits);
    
    //hit covariance matrix
    GlobalParameterMatrix Vinv = GlobalParameterMatrix::Zero(nhitparms, nhitparms);
    //process noise matrix (MS+stochastic energy loss)
    GlobalParameterMatrix Qinv = GlobalParameterMatrix::Zero(nmomparms, nmomparms);
    GlobalParameterMatrix Qinvpos = GlobalParameterMatrix::Zero(nposparms, nposparms);
    
    //hit residuals
    GlobalParameterVector dy0 = GlobalParameterVector::Zero(nhitparms);
    //momentum kink residuals
    GlobalParameterVector dx0mom = GlobalParameterVector::Zero(nmomparms);
    //position kink residuals
    GlobalParameterVector dx0pos = GlobalParameterVector::Zero(nposparms);
    
//     MatrixXd P = MatrixXd::Zero(2*(nhits-1), 5*nhits);
    
    globalidxv.clear();
    globalidxv.resize(npars, 0);
    
    nParms = npars;
    tree->SetBranchAddress("globalidxv", globalidxv.data());
    
    TrajectoryStateOnSurface currtsos;
    
    bool valid = true;
    unsigned int ntotalhitdim = 0;
    unsigned int alignmentidx = 0;
    unsigned int bfieldidx = 0;
    unsigned int elossidx = 0;
    for (unsigned int i = 0; i < tms.size(); ++i)
    {
      
      TrajectoryMeasurement const& tm = tms[i];
      auto const& hit = tm.recHit();
//       TrajectoryStateOnSurface const& backpredtsos = tm.backwardPredictedState();
//       TrajectoryStateOnSurface const& fwdpredtsos = tm.forwardPredictedState();
//       TrajectoryStateOnSurface updtsos = combiner(backpredtsos, fwdpredtsos);
      TrajectoryStateOnSurface const& updtsos = tm.updatedState();
      const GeomDet *detectorG = globalGeometry->idToDet(hit->geographicalId());
            
      if (!updtsos.isValid()) {
        std::cout << "Abort: updtsos invalid" << std::endl;
        valid = false;
        break;
      }
      
//       TrajectoryStateOnSurface backupdtsos(backpredtsos);
      
      //hit information
      if (hit->isValid()) {
        ntotalhitdim += std::min(hit->dimension(), 2);
        //update state if hit is valid
//         backupdtsos = updator.update(backpredtsos, *hit);
//         if (!backupdtsos.isValid()) {
//           std::cout << "Abort: hit update failed" << std::endl;
//           valid = false;
//           break;
//         }
        
        //fill x residual
        dy0(2*i) = hit->localPosition().x() - updtsos.localPosition().x();
        
        //check for 2d hits
        if (hit->dimension()>1) {
          //fill y residual
          dy0(2*i+1) = hit->localPosition().y() - updtsos.localPosition().y();
          
          //compute 2x2 covariance matrix and invert
          Matrix2d iV;
          iV << hit->localPositionError().xx(), hit->localPositionError().xy(),
                hit->localPositionError().xy(), hit->localPositionError().yy();
          Vinv.block<2,2>(2*i, 2*i) = iV.inverse();
        }
        else {
          //covariance is 1x1
          Vinv(2*i, 2*i) = 1./hit->localPositionError().xx();
        }
      }
      
//       std::cout << "dy0:" << std::endl;
//       std::cout << dy0.segment<2>(2*i) << std::endl;
//       std::cout << "Vinv:" << std::endl;
//       std::cout << Vinv.block<2,2>(2*i, 2*i) << std::endl;
      
      JacobianCurvilinearToLocal h(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
      const AlgebraicMatrix55 &jach = h.jacobian();
      //efficient assignment from SMatrix using Eigen::Map
      Map<const Matrix<double, 5, 5, RowMajor> > jacheig(jach.Array());
      Hh.block<2,5>(2*i, 5*i) = jacheig.bottomRows<2>();
      
//       std::cout << "H:" << std::endl;
//       std::cout << H.block<5,5>(5*i, 5*i) << std::endl;
      
      //fill alignment jacobian
      bool ispixel = GeomDetEnumerators::isTrackerPixel(detectorG->subDetector());
      //2d alignment correction for pixel hits, otherwise 1d
      A(2*i, alignmentidx) = 1.;
      const unsigned int xglobalidx = detidparms.at(std::make_pair(0,hit->geographicalId()));
      globalidxv[alignmentidx] = xglobalidx;
      alignmentidx++;
      if (ispixel) {
        A(2*i+1, alignmentidx) = 1.;
        const unsigned int yglobalidx = detidparms.at(std::make_pair(1,hit->geographicalId()));
        globalidxv[alignmentidx] = yglobalidx;
        alignmentidx++;
      }

      
      if (i >0) {
        //fill jacobians for nominal state
        Hmom.block<3,5>(3*(i-1), 5*i) = jacheig.topRows<3>();
        Hpos.block<2,5>(2*(i-1), 5*i) = jacheig.bottomRows<2>();
        
        //propagate the previous updated state outside-in as in the smoother to compute the path length
        //use the geometrical propagator since the material effects are dealt with separately
        auto const& propresult = rPropagator.geometricalPropagator().propagateWithPath(currtsos, updtsos.surface());
        if (!propresult.first.isValid()) {
          std::cout << "Abort: propagation failed" << std::endl;
          valid = false;
          break;
        }
        TrajectoryStateOnSurface proptsos = propresult.first;
        const double s = propresult.second;
        
        //compute transport jacobian
        AnalyticalCurvilinearJacobian curvjac(currtsos.globalParameters(), proptsos.globalParameters().position(), proptsos.globalParameters().momentum(), s);
        const AlgebraicMatrix55 &jacF = curvjac.jacobian();
        F.block<5,5>(5*(i-1), 5*(i-1)) = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array());
       
        //analytic jacobian wrt magnitude of magnetic field
        //TODO should we parameterize with respect to z-component instead?
        //extending derivation from CMS NOTE 2006/001
        const Vector3d b(currtsos.globalParameters().magneticFieldInInverseGeV().x(),
                           currtsos.globalParameters().magneticFieldInInverseGeV().y(),
                           currtsos.globalParameters().magneticFieldInInverseGeV().z());
        double magb = b.norm();
        const Vector3d h = b.normalized();

        const Vector3d p0(currtsos.globalParameters().momentum().x(),
                            currtsos.globalParameters().momentum().y(),
                            currtsos.globalParameters().momentum().z());
        const Vector3d p1(proptsos.globalParameters().momentum().x(),
                            proptsos.globalParameters().momentum().y(),
                            proptsos.globalParameters().momentum().z());
        const Vector3d T0 = p0.normalized();
        double p = p1.norm();
        const Vector3d T = p1.normalized();
        double q = currtsos.charge();
        
        const Vector3d N0 = h.cross(T0).normalized();
        const double alpha = h.cross(T).norm();
        const double gamma = h.transpose()*T;

        //this is printed from sympy.printing.cxxcode together with sympy.cse for automatic substitution of common expressions
        auto const xf0 = q*s/p;
        auto const xf1 = magb*xf0;
        auto const xf2 = std::cos(xf1);
        auto const xf3 = 1.0/magb;
        auto const xf4 = s*xf3;
        auto const xf5 = std::sin(xf1);
        auto const xf6 = p/q;
        auto const xf7 = xf6/std::pow(magb, 2);
        auto const xf8 = T0;
        auto const xf9 = xf8;
        auto const xf10 = alpha*xf5;
        auto const xf11 = 1 - xf2;
        auto const xf12 = N0;
        auto const xf13 = xf12;
        auto const xf14 = xf0*xf2;
        auto const xf15 = h;
        auto const xf16 = xf15;
        auto const xf17 = (xf2*xf4 - xf5*xf7)*xf9 + (alpha*xf11*xf7 - xf10*xf4)*xf13 + (-gamma*xf3*xf6*(-xf0 + xf14) + gamma*xf7*(-xf1 + xf5))*xf16;
        auto const xf18 = (-xf10)*xf12.transpose() + xf2*(xf8.transpose()) + (gamma*xf11)*(xf15.transpose());
        auto const xf19 = xf0*xf5;
        auto const resf0 = xf17 - xf17*xf18*xf17;
        auto const resf1 = (-p)*((-xf19)*xf9 + (-alpha*xf14)*xf13 + (gamma*xf19)*xf16)*xf18*xf17 + xf17;

        const Vector3d dMdB = resf0;
        const Vector3d dPdB = resf1;

        Vector6d dFglobal;
        dFglobal.head<3>() = dMdB;
        dFglobal.tail<3>() = dPdB;
        
        //convert to curvilinear
        JacobianCartesianToCurvilinear cart2curv(proptsos.globalParameters());
        const AlgebraicMatrix56& cart2curvjacs = cart2curv.jacobian();
        const Map<const Matrix<double, 5, 6, RowMajor> > cart2curvjac(cart2curvjacs.Array());
        
        //compute final jacobian (and convert to Tesla)
        dF.block<5,1>(5*(i-1), bfieldidx) = 2.99792458e-3*cart2curvjac*dFglobal;
        
//         std::cout << "dF = " << dF.block<5,1>(5*i, bfieldidx) << std::endl;
        
        const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(2,hit->geographicalId()));
        globalidxv[nparsAlignment + bfieldidx] = bfieldglobalidx;
        bfieldidx++;
        
        //compute the inverse energy loss jacobian
        //full analytic energy loss jacobian (gross)
        //n.b this is the jacobian in LOCAL parameters (so E multiplies to the left of H s.t the total projection is E*Hprop*F)
        const double m2 = pow(rPropagator.materialEffectsUpdator().mass(), 2);  // use mass hypothesis from constructor
        constexpr double emass = 0.511e-3;
        constexpr double poti = 16.e-9 * 10.75;                 // = 16 eV * Z**0.9, for Si Z=14
        const double eplasma = 28.816e-9 * sqrt(2.33 * 0.498);  // 28.816 eV * sqrt(rho*(Z/A)) for Si
        const double qop = proptsos.localParameters().qbp();
        const double dxdz = proptsos.localParameters().dxdz();
        const double dydz = proptsos.localParameters().dydz();
        const double xi = proptsos.surface().mediumProperties().xi();
//         printf("xi = %5e\n", xi);
        
        //this is printed from sympy.printing.cxxcode together with sympy.cse for automatic substitution of common expressions
        const double x0 = (((qop) > 0) - ((qop) < 0));
        const double x1 = std::pow(qop, 2);
        const double x2 = 1.0/x1;
        const double x3 = x0*x2;
        const double x4 = m2*x1 + 1;
        const double x5 = 1.0/x4;
        const double x6 = std::pow(poti, 2);
        const double x7 = 1.0/x6;
        const double x8 = std::fabs(qop);
        const double x9 = std::pow(emass, 2);
        const double x10 = x8*x9;
        const double x11 = std::sqrt(x4);
        const double x12 = 2*emass;
        const double x13 = m2*x8 + x10 + x11*x12;
        const double x14 = 1.0/x13;
        const double x15 = x10*x14*x7;
        const double x16 = 4*x2;
        const double x17 = -2*std::log(eplasma/poti) + std::log(x15*x16) + 1;
        const double x18 = x17*x4 - 2;
        const double x19 = std::pow(x5, -1.0/2.0);
        const double x20 = std::sqrt(std::pow(dxdz, 2) + std::pow(dydz, 2) + 1);
        const double x21 = x19*x20*xi;
        const double x22 = x18*x21;
        const double x23 = m2*qop;
        const double x24 = x0/std::pow(x22 + x0/qop, 2);
        const double x25 = x18*x19*x24;
        const double x26 = x25*xi/x20;
        const double res_0 = x24*(-x21*((1.0/4.0)*x1*x13*x4*x6*(x10*x16*x7*(-m2*x0 - x0*x9 - x12*x23/x11)/std::pow(x13, 2) + 4*x14*x3*x7*x9 - 8*x15/std::pow(qop, 3))/(x8*x9) + 2*x17*x23) - x22*x23*x5 + x3);
        const double res_1 = -dxdz*x26;
        const double res_2 = -dydz*x26;
        const double res_3 = -x20*x25;

        E(3*(i-1), 3*(i-1)) = res_0;
        E(3*(i-1), 3*(i-1)+1) = res_1;
        E(3*(i-1), 3*(i-1)+2) = res_2;
        E(3*(i-1)+1, 3*(i-1)+1) = 1.;
        E(3*(i-1)+2, 3*(i-1)+2) = 1.;
        
//         std::cout << "Eloss 0,0 = " << E(5*i,5*i) << std::endl;
        
        //derivative of the energy loss with respect to the energy loss parameter xi
        dE(3*(i-1), elossidx) = res_3;
        const unsigned int elossglobalidx = detidparms.at(std::make_pair(3,hit->geographicalId()));
        globalidxv[nparsAlignment + nparsBfield + elossidx] = elossglobalidx;
        elossidx++;
       
        //apply material effects in reverse
        //zero local errors to directly access the process noise matrix
        //and force the state to be on the correct side
        proptsos.update(proptsos.localParameters(),
                       LocalTrajectoryError(0.,0.,0.,0.,0.),
                       proptsos.surface(),
                       proptsos.magneticField(),
                       SurfaceSideDefinition::afterSurface);
        
        //apply the state update from the material effects (in reverse)
        bool ok = rPropagator.materialEffectsUpdator().updateStateInPlace(proptsos, rpropdir);
        if (!ok) {
          std::cout << "Abort: material update failed" << std::endl;
          valid = false;
          break;
        }
       
        //get the process noise matrix
        AlgebraicMatrix55 const Qmat = proptsos.localError().matrix();
        Map<const Matrix<double, 5, 5, RowMajor> >iQ(Qmat.Array());
        //Q is 3x3 in the upper left block because there is no displacement on thin scattering layers
        //so invert the upper 3x3 block
        //(The zero-displacement constraint is implemented with Lagrange multipliers)
        Qinv.block<3,3>(3*(i-1), 3*(i-1)) = iQ.topLeftCorner<3,3>().inverse();
        
        //zero displacement on thin scattering layer approximated with small uncertainty
        const double epsxy = 1e-5; //0.1um
        Qinvpos(2*(i-1), 2*(i-1)) = 1./epsxy/epsxy;
        Qinvpos(2*(i-1)+1, 2*(i-1)+1) = 1./epsxy/epsxy;
        
        //fill jacobians for backwards propagated state
        JacobianCurvilinearToLocal hprop(proptsos.surface(), proptsos.localParameters(), *proptsos.magneticField());
        const AlgebraicMatrix55 &jachprop = hprop.jacobian();
        //efficient assignment from SMatrix using Eigen::Map
        Map<const Matrix<double, 5, 5, RowMajor> > jachpropeig(jachprop.Array());
        Hpropmom.block<3,5>(3*(i-1), 5*(i-1)) = jachpropeig.topRows<3>();
        Hproppos.block<2,5>(2*(i-1), 5*(i-1)) = jachpropeig.bottomRows<2>();
        
        //compute kink residuals
        AlgebraicVector5 const& idx0 = updtsos.localParameters().vector() - proptsos.localParameters().vector();        
        Map<const Vector5d> idx0eig(idx0.Array());
        dx0mom.segment<3>(3*(i-1)) = idx0eig.head<3>();
        dx0pos.segment<2>(2*(i-1)) = idx0eig.tail<2>();
                
      }
      currtsos = updtsos;
      
//       std::cout << "meas: " << i << std::endl;
//       std::cout << "dy0:" << std::endl;
//       std::cout << dy0.segment<2>(2*i) << std::endl;
//       std::cout << "Vinv:" << std::endl;
//       std::cout << Vinv.block<2,2>(2*i, 2*i) << std::endl;
//       std::cout << "Hh:" << std::endl;
//       std::cout << Hh.block<2,5>(2*i, 5*i) << std::endl;
//       if (i>0) {
//         std::cout << "dx0mom" << std::endl;
//         std::cout << dx0mom.segment<3>(3*(i-1)) << std::endl;
//         std::cout << "dx0pos" << std::endl;
//         std::cout << dx0pos.segment<2>(2*(i-1)) << std::endl;
//         std::cout << "Qinv" << std::endl;        
//         std::cout << Qinv.block<3,3>(3*(i-1), 3*(i-1)) << std::endl;
//         std::cout << "Qinvpos" << std::endl;        
//         std::cout << Qinvpos.block<2,2>(2*(i-1), 2*(i-1)) << std::endl;
//         std::cout << "F" << std::endl;        
//         std::cout << F.block<3,3>(3*(i-1), 3*(i-1)) << std::endl;
//         std::cout << "Hmom:" << std::endl;
//         std::cout << Hmom.block<3,5>(3*(i-1), 5*i) << std::endl;
//         std::cout << "Hpos:" << std::endl;
//         std::cout << Hpos.block<2,5>(2*(i-1), 5*i) << std::endl;     
//         std::cout << "Hpropmom:" << std::endl;
//         std::cout << Hpropmom.block<3,5>(3*(i-1), 5*(i-1)) << std::endl;
//         std::cout << "Hproppos:" << std::endl;
//         std::cout << Hproppos.block<2,5>(2*(i-1), 5*(i-1)) << std::endl;   
//       }
//       std::cout << "updtsos local parameters" << std::endl;
//       std::cout << updtsos.localParameters().vector() << std::endl;
//       std::cout << "xi" << std::endl;
//       std::cout << updtsos.surface().mediumProperties().xi() << std::endl;
      
    }
    
    if (!valid) {
      continue;
    }
    
    auto const& refpoint = track.referencePoint();
    auto const& trackmom = track.momentum();
    
    const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
    const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z());
    
    //propagate track from reference point to first measurement to compute the pathlength, and then compute
    //the transport jacobian corresponding to the reverse propagation
    const TrajectoryStateOnSurface &innertsos = tms.back().updatedState();
    FreeTrajectoryState refFts(refpos, refmom, track.charge(), innertsos.magneticField());
    auto const& propresult = fPropagator.geometricalPropagator().propagateWithPath(refFts, innertsos.surface());
    if (!propresult.first.isValid()) {
      std::cout << "Abort: propagation from reference point failed" << std::endl;
      continue;
    }
    AnalyticalCurvilinearJacobian curvjac(propresult.first.globalParameters(), refFts.position(), refFts.momentum(), -propresult.second);
    const AlgebraicMatrix55& jacF = curvjac.jacobian();
    Fref.rightCols<5>() = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array());

    //now do the expensive calculations and fill outputs
    
    gradv.clear();
    jacrefv.clear();

    gradv.resize(npars,0.);
    jacrefv.resize(5*npars, 0.);
    
    nJacRef = 5*npars;
    tree->SetBranchAddress("gradv", gradv.data());
    tree->SetBranchAddress("jacrefv", jacrefv.data());
    
    //eigen representation of the underlying vector storage
    Map<VectorXf> gradout(gradv.data(), npars);
    Map<Matrix<float, 5, Dynamic, RowMajor> > jacrefout(jacrefv.data(), 5, npars);
    
    //hessian is not stored directly so it's a standard Eigen Matrix
    VectorXd grad = VectorXd::Zero(npars);
    MatrixXd hess = MatrixXd::Zero(npars, npars);
    Matrix<double, 5, Dynamic> jacref = Matrix<double, 5, Dynamic>::Zero(5, npars);
    
    //compute covariance explicitly to simplify below expressions
    const MatrixXd C = (2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F) + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom) + 2*Hh.transpose()*Vinv*Hh).ldlt().solve(MatrixXd::Identity(nstateparms,nstateparms));
    
    //compute shift in parameters at reference point from matrix model given hit and kink residuals
    //(but for nominal values of the model correction
    Vector5d dxRef = -Fref*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0);
    
    //fill output with corrected state and covariance at reference point
    refParms.fill(0.);
    refCov.fill(0.);
    const AlgebraicVector5& refVec = track.parameters();
    Map<Vector5f>(refParms.data()) = (Map<const Vector5d>(refVec.Array()) + dxRef).cast<float>();
    Map<Matrix<float, 5, 5, RowMajor> >(refCov.data()).triangularView<Upper>() = (2.*Fref*C*Fref.transpose()).cast<float>().triangularView<Upper>();
    
    //set jacobian for reference point parameters
    jacref.leftCols(nparsAlignment) = -2*Fref*C*Hh.transpose()*Vinv*A;

    jacref.block<5, Dynamic>(0, nparsAlignment, 5, nparsBfield) = Fref*(2*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 2*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF);
    
    jacref.rightCols(nparsEloss) = 2*Fref*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE;
    
    jacrefout = jacref.cast<float>();
    
//     std::cout << "jacref:" << std::endl;
//     std::cout << jacref << std::endl;
    
    //set gradients
    grad.head(nparsAlignment) = -2*A.transpose()*Vinv*(Hh*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dy0) - 4*A.transpose()*Vinv*Hh*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(-(Hpos - Hproppos*F)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0pos) - 4*A.transpose()*Vinv*Hh*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) + 4*A.transpose()*Vinv*Hh*C.transpose()*Hh.transpose()*Vinv*(Hh*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dy0);

    grad.segment(nparsAlignment,nparsBfield) = -2*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) + 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(-(Hpos - Hproppos*F)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0pos) + 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) - 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*(Hh*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dy0) - 2*dF.transpose()*Hproppos.transpose()*Qinvpos*(-(Hpos - Hproppos*F)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0pos) + 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(-(Hpos - Hproppos*F)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0pos) + 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) - 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*Hh.transpose()*Vinv*(Hh*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dy0);
    
    grad.tail(nparsEloss) = -2*dE.transpose()*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) + 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(-(Hpos - Hproppos*F)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0pos) + 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-(-E*Hpropmom*F + Hmom)*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dx0mom) - 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*(Hh*C*(2*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*dx0pos + 2*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dx0mom - 2*Hh.transpose()*Vinv*dy0) + dy0);
        
    gradout = grad.cast<float>();
    
    //fill hessian (diagonal blocks, upper triangular part only)
    hess.topLeftCorner(nparsAlignment, nparsAlignment).triangularView<Upper>() = (2*A.transpose()*Vinv*A - 4*A.transpose()*Vinv*Hh*C*Hh.transpose()*Vinv*A + 8*A.transpose()*Vinv*Hh*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*Hh.transpose()*Vinv*A + 8*A.transpose()*Vinv*Hh*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A - 4*A.transpose()*Vinv*Hh*C.transpose()*Hh.transpose()*Vinv*A + 8*A.transpose()*Vinv*Hh*C.transpose()*Hh.transpose()*Vinv*Hh*C*Hh.transpose()*Vinv*A).triangularView<Upper>();
    
    hess.block(nparsAlignment, nparsAlignment, nparsBfield, nparsBfield).triangularView<Upper>() = (-4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF - 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 2*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*E*Hpropmom*dF - 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF - 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 2*dF.transpose()*Hproppos.transpose()*Qinvpos*Hproppos*dF).triangularView<Upper>();
    
    hess.bottomRightCorner(nparsEloss,nparsEloss).triangularView<Upper>() = (-4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE - 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*dE + 2*dE.transpose()*Qinv*dE).triangularView<Upper>();
    
    //fill hessian off-diagonal blocks (upper triangular part)
    hess.transpose().block(nparsAlignment, 0, nparsBfield, nparsAlignment) = 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A + 4*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hpropmom.transpose()*E.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*Hh.transpose()*Vinv*A + 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A + 4*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*Hh.transpose()*Vinv*A - 8*dF.transpose()*Hproppos.transpose()*Qinvpos*(Hpos - Hproppos*F)*C.transpose()*Hh.transpose()*Vinv*Hh*C*Hh.transpose()*Vinv*A;
    
    hess.transpose().block(nparsAlignment+nparsBfield, 0, nparsEloss, nparsAlignment) = 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*Hh.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*Hh.transpose()*Vinv*A + 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*Hh.transpose()*Vinv*A;
    
    //careful this is easy to screw up because it is "accidentally" symmetric
    hess.transpose().block(nparsAlignment+nparsBfield, nparsAlignment, nparsEloss, nparsBfield) = -4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF - 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*(Hpos - Hproppos*F)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*(-E*Hpropmom*F + Hmom)*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF - 4*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hproppos.transpose() + Hpos.transpose())*Qinvpos*Hproppos*dF + 8*dE.transpose()*Qinv*(-E*Hpropmom*F + Hmom)*C.transpose()*Hh.transpose()*Vinv*Hh*C*(-F.transpose()*Hpropmom.transpose()*E.transpose() + Hmom.transpose())*Qinv*E*Hpropmom*dF + 2*dE.transpose()*Qinv*E*Hpropmom*dF;
    
//     std::cout << "hess debug" << std::endl;
//     std::cout << "original cov" << std::endl;
//     std::cout << tms[nhits-1].updatedState().curvilinearError().matrix() << std::endl;
//     std::cout << "recomputed cov" << std::endl;
//     std::cout << 2*C.bottomRightCorner<5,5>() << std::endl;
    
    
    //fill packed hessian and indices
    const unsigned int nsym = npars*(1+npars)/2;
    hesspackedv.clear();    
    hesspackedv.resize(nsym, 0.);
    
    nSym =nsym;
    tree->SetBranchAddress("hesspackedv", hesspackedv.data());
    
    Map<VectorXf> hesspacked(hesspackedv.data(), nsym);
    const Map<const VectorXu> globalidx(globalidxv.data(), npars);

    unsigned int packedidx = 0;
    for (unsigned int ipar = 0; ipar < npars; ++ipar) {
      const unsigned int segmentsize = npars - ipar;
      hesspacked.segment(packedidx, segmentsize) = hess.block<1, Dynamic>(ipar, ipar, 1, segmentsize).cast<float>();
      packedidx += segmentsize;
    }

    tree->Fill();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ResidualGlobalCorrectionMaker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void ResidualGlobalCorrectionMaker::endJob()
{
  fout->cd();
  fout->Write();
  fout->Close();
}

// ------------ method called when starting to processes a run  ------------

void 
ResidualGlobalCorrectionMaker::beginRun(edm::Run const& run, edm::EventSetup const& es)
{
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  es.get<GlobalTrackingGeometryRecord>().get(globalGeometry);
  
  detidparms.clear();
  
  std::set<std::pair<int, DetId> > parmset;
  
  for (const GeomDet* det : globalGeometry->dets()) {
    if (!det) {
      continue;
    }
    if (GeomDetEnumerators::isTracker(det->subDetector())) {
      //always have parameters for local x alignment, bfield, and e-loss
      parmset.emplace(0, det->geographicalId());
      parmset.emplace(2, det->geographicalId());
      parmset.emplace(3, det->geographicalId());
      if (GeomDetEnumerators::isTrackerPixel(det->subDetector())) {
        //local y alignment parameters only for pixels for now
        parmset.emplace(1, det->geographicalId());
      }
    }
  }
  
  TFile *runfout = new TFile("trackTreeGradsParmInfo.root", "RECREATE");
  TTree *runtree = new TTree("tree", "tree");
  
  unsigned int iidx;
  int parmtype;
  unsigned int rawdetid;
  int subdet;
  int layer;
  float x;
  float y;
  float z;
  float eta;
  float phi;
  float rho;

  runtree->Branch("iidx", &iidx);
  runtree->Branch("parmtype", &parmtype);
  runtree->Branch("rawdetid", &rawdetid);
  runtree->Branch("subdet", &subdet);
  runtree->Branch("layer", &layer);
  runtree->Branch("x", &x);
  runtree->Branch("y", &y);
  runtree->Branch("z", &z);
  runtree->Branch("eta", &eta);
  runtree->Branch("phi", &phi);
  runtree->Branch("rho", &rho);
  
  unsigned int globalidx = 0;
  for (const auto& key: parmset) {
    
    //fill info
    const DetId& detid = key.second;
    const GeomDet* det = globalGeometry->idToDet(detid);
    
    layer = 0;
//     int subdet = det->subDetector();
//     float eta = det->surface().position().eta();

    if (det->subDetector() == GeomDetEnumerators::PixelBarrel)
    {
      PXBDetId detid(det->geographicalId());
      layer = detid.layer();
    }
    else if (det->subDetector() == GeomDetEnumerators::PixelEndcap)
    {
      PXFDetId detid(det->geographicalId());
      layer = -1 * (detid.side() == 1) * detid.disk() + (detid.side() == 2) * detid.disk();
    }
    else if (det->subDetector() == GeomDetEnumerators::TIB)
    {
      TIBDetId detid(det->geographicalId());
      layer = detid.layer();
    }
    else if (det->subDetector() == GeomDetEnumerators::TOB)
    {
      TOBDetId detid(det->geographicalId());
      layer = detid.layer();
    }
    else if (det->subDetector() == GeomDetEnumerators::TID)
    {
      TIDDetId detid(det->geographicalId());
      layer = -1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel();

    }
    else if (det->subDetector() == GeomDetEnumerators::TEC)
    {
      TECDetId detid(det->geographicalId());
      layer = -1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel();
    }
    
//     ParmInfo parminfo;
//     parminfo.parmtype = key.first;
//     parminfo.subdet = det->subDetector();
//     parminfo.layer = layer;
//     parminfo.x = det->surface().position().x();
//     parminfo.y = det->surface().position().y();
//     parminfo.z = det->surface().position().z();
//     parminfo.eta = det->surface().position().eta();
//     parminfo.phi = det->surface().position().phi();
//     parminfo.rho = det->surface().position().perp();

    iidx = globalidx;
    parmtype = key.first;
    rawdetid = detid;
    subdet = det->subDetector();
    //layer already set above
    x = det->surface().position().x();
    y = det->surface().position().y();
    z = det->surface().position().z();
    eta = det->surface().position().eta();
    phi = det->surface().position().phi();
    rho = det->surface().position().perp();
    
    //fill map
    detidparms.emplace(key, globalidx);
    globalidx++;
    
    runtree->Fill();
  }
  
  runfout->Write();
  runfout->Close();
  
//   std::sort(detidparms.begin(), detidparms.end());
  std::cout << "nglobalparms = " << detidparms.size() << std::endl;
    
  
}


// ------------ method called when ending the processing of a run  ------------
/*
void 
HitAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
HitAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
HitAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ResidualGlobalCorrectionMaker::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

template <unsigned int D>
AlgebraicVector5 ResidualGlobalCorrectionMaker::lupdate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit) {
  typedef typename AlgebraicROOTObject<D, 5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5, D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D, D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;
  using ROOT::Math::SMatrixNoInit;

  auto&& x = tsos.localParameters().vector();
  auto&& C = tsos.localError().matrix();

  // projection matrix (assume element of "H" to be just 0 or 1)
  ProjectMatrix<double, 5, D> pf;

  // Measurement matrix
  VecD r, rMeas;
  SMatDD V(SMatrixNoInit{}), VMeas(SMatrixNoInit{});

  KfComponentsHolder holder;
  holder.template setup<D>(&r, &V, &pf, &rMeas, &VMeas, x, C);
  aRecHit.getKfComponents(holder);

  r -= rMeas;

  // and covariance matrix of residuals
  SMatDD R = V + VMeas;
  bool ok = invertPosDefMatrix(R);
  if (!ok) {
    return AlgebraicVector5();
  }

  // Compute Kalman gain matrix
  AlgebraicMatrix55 M = AlgebraicMatrixID();
  Mat5D K = C * pf.project(R);
  pf.projectAndSubtractFrom(M, K);

  // Compute local filtered state vector
  AlgebraicVector5 fsvdiff = K * r;
  return fsvdiff;
}

AlgebraicVector5 ResidualGlobalCorrectionMaker::update(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit) {
  switch (aRecHit.dimension()) {
    case 1:
      return lupdate<1>(tsos, aRecHit);
    case 2:
      return lupdate<2>(tsos, aRecHit);
    case 3:
      return lupdate<3>(tsos, aRecHit);
    case 4:
      return lupdate<4>(tsos, aRecHit);
    case 5:
      return lupdate<5>(tsos, aRecHit);
  }
  return AlgebraicVector5();
}

//define this as a plug-in
DEFINE_FWK_MODULE(ResidualGlobalCorrectionMaker);
