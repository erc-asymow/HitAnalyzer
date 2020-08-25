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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
// #include "DataFormats/SiStripDetId/interface/TIDDetId.h"
// #include "DataFormats/SiStripDetId/interface/TIBDetId.h"
// #include "DataFormats/SiStripDetId/interface/TOBDetId.h"
// #include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/TkRadialStripTopology.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
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
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"





// #include "../interface/OffsetMagneticField.h"
// #include "../interface/ParmInfo.h"


#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "functions.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/AutoDiff>
#include<Eigen/StdVector>
#include <iostream>
#include <functional>

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

template<typename T>
using evector = std::vector<T, Eigen::aligned_allocator<T>>;


namespace std{
  template <>
  struct hash<std::pair<unsigned int, unsigned int>> {
      size_t operator() (const std::pair<unsigned int, unsigned int>& s) const {
        unsigned long long rawkey = s.first;
        rawkey = rawkey << 32;
        rawkey += s.second;
        return std::hash<unsigned long long>()(rawkey); 
      }
  };
}


//double active scalar for autodiff grad+hessian
//template arguments are datatype, and size of gradient
//In this form the gradients are null length unless/until the variable
//is initialized with init_twice_active_var (though the corresponding
//memory is still used on the stack)
template<typename T, int N>
using AANT = AutoDiffScalar<Matrix<AutoDiffScalar<Matrix<T, N, 1>>, Dynamic, 1, 0, N, 1>>;

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

  Matrix<double, 5, 1> bfieldJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s) const;
                                                             
  Matrix<double, 5, 6> materialEffectsJacobian(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator);
  
  std::array<Matrix<double, 5, 5>, 5> processNoiseJacobians(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const;
  
  template <unsigned int D>
  AlgebraicVector5 lupdate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  AlgebraicVector5 update(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  template <typename T>
  void init_twice_active_var(T &ad, const unsigned int d_num, const unsigned int idx) const;
  
  template <typename T>
  void init_twice_active_null(T &ad, const unsigned int d_num) const;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;
//   edm::EDGetTokenT<TrajTrackAssociationCollection> inputTrack_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrack_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrackOrig_;
  edm::EDGetTokenT<std::vector<int> > inputIndices_;
  edm::EDGetTokenT<reco::BeamSpot> inputBs_;

//   TFile *fout;
  TTree *tree;
  TTree *runtree;
  TTree *gradtree;
  TTree *hesstree;

  float trackEta;
  float trackPhi;
  float trackPt;
  float trackPtErr;
  float trackCharge;

  float normalizedChi2;
  
  float genPt;
  float genEta;
  float genPhi;
  float genCharge;
  
  unsigned int nHits;
  unsigned int nValidHits;
  unsigned int nValidPixelHits;
  unsigned int nParms;
  unsigned int nJacRef;
  unsigned int nSym;
  
  float gradmax;
  float hessmax;
  
  std::array<float, 5> trackOrigParms;
  std::array<float, 25> trackOrigCov;
  
  
  std::array<float, 5> trackParms;
  std::array<float, 25> trackCov;
  
  std::array<float, 5> refParms_iter0;
  std::array<float, 25> refCov_iter0;

  std::array<float, 5> refParms_iter2;
  std::array<float, 25> refCov_iter2;
  
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
  
  std::vector<double> gradagg;
  
  std::unordered_map<std::pair<unsigned int, unsigned int>, double> hessaggsparse;
  
  bool fitFromGenParms_;
  bool fillTrackTree_;
  
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
//   inputTrack_ = consumes<TrajTrackAssociationCollection>(edm::InputTag("TrackRefitter"));
  inputTrack_ = consumes<reco::TrackCollection>(edm::InputTag("TrackRefitter"));
  inputIndices_ = consumes<std::vector<int> >(edm::InputTag("TrackRefitter"));
  inputTrackOrig_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  inputBs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  
  fitFromGenParms_ = iConfig.getParameter<bool>("fitFromGenParms");
  fillTrackTree_ = iConfig.getParameter<bool>("fillTrackTree");


//   fout = new TFile("trackTreeGrads.root", "RECREATE");
//   fout = new TFile("trackTreeGradsdebug.root", "RECREATE");
//   fout = new TFile("trackTreeGrads.root", "RECREATE");
  //TODO this needs a newer root version
//   fout->SetCompressionAlgorithm(ROOT::kLZ4);
//   fout->SetCompressionLevel(3);
  
  edm::Service<TFileService> fs;
  
//   tree = new TTree("tree", "tree");
  
  runtree = fs->make<TTree>("runtree","");
  gradtree = fs->make<TTree>("gradtree","");
  hesstree = fs->make<TTree>("hesstree","");
  
  
  if (fillTrackTree_) {
    tree = fs->make<TTree>("tree","");
    const int basketSize = 4*1024*1024;
    tree->SetAutoFlush(0);
    
    tree->Branch("trackPt", &trackPt, basketSize);
    tree->Branch("trackPtErr", &trackPtErr, basketSize);
    tree->Branch("trackEta", &trackEta, basketSize);
    tree->Branch("trackPhi", &trackPhi, basketSize);
    tree->Branch("trackCharge", &trackCharge, basketSize);
    //workaround for older ROOT version inability to store std::array automatically
  //   tree->Branch("trackOrigParms", trackOrigParms.data(), "trackOrigParms[5]/F", basketSize);
  //   tree->Branch("trackOrigCov", trackOrigCov.data(), "trackOrigCov[25]/F", basketSize);
    tree->Branch("trackParms", trackParms.data(), "trackParms[5]/F", basketSize);
    tree->Branch("trackCov", trackCov.data(), "trackCov[25]/F", basketSize);
    
    tree->Branch("refParms_iter0", refParms_iter0.data(), "refParms_iter0[5]/F", basketSize);
    tree->Branch("refCov_iter0", refCov_iter0.data(), "refCov_iter0[25]/F", basketSize);
  //   tree->Branch("refParms_iter2", refParms_iter2.data(), "refParms_iter2[5]/F", basketSize);
  //   tree->Branch("refCov_iter2", refCov_iter2.data(), "refCov_iter2[25]/F", basketSize);  
    
    tree->Branch("refParms", refParms.data(), "refParms[5]/F", basketSize);
    tree->Branch("refCov", refCov.data(), "refCov[25]/F", basketSize);
    tree->Branch("genParms", genParms.data(), "genParms[5]/F", basketSize);

    tree->Branch("genPt", &genPt, basketSize);
    tree->Branch("genEta", &genEta, basketSize);
    tree->Branch("genPhi", &genPhi, basketSize);
    tree->Branch("genCharge", &genCharge, basketSize);
    
    tree->Branch("normalizedChi2", &normalizedChi2, basketSize);
    
    tree->Branch("nHits", &nHits, basketSize);
    tree->Branch("nValidHits", &nValidHits, basketSize);
    tree->Branch("nValidPixelHits", &nValidPixelHits, basketSize);
    tree->Branch("nParms", &nParms, basketSize);
    tree->Branch("nJacRef", &nJacRef, basketSize);
    
  //   tree->Branch("gradv", gradv.data(), "gradv[nParms]/F", basketSize);
    tree->Branch("globalidxv", globalidxv.data(), "globalidxv[nParms]/i", basketSize);
    tree->Branch("jacrefv",jacrefv.data(),"jacrefv[nJacRef]/F", basketSize);
    
  //   tree->Branch("nSym", &nSym, basketSize);
    
  //   tree->Branch("hesspackedv", hesspackedv.data(), "hesspackedv[nSym]/F", basketSize);
    
    tree->Branch("run", &run);
    tree->Branch("lumi", &lumi);
    tree->Branch("event", &event);
    
    tree->Branch("gradmax", &gradmax);
    tree->Branch("hessmax", &hessmax);
  }

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
  
  const bool dogen = fitFromGenParms_;
  
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
  
  ESHandle<Propagator> thePropagator;
  iSetup.get<TrackingComponentsRecord>().get("RungeKuttaTrackerPropagator", thePropagator);
  
  
//   Handle<TrajTrackAssociationCollection> trackH;
//   Handle<reco::TrackCollection> trackH;
//   iEvent.getByToken(inputTrack_, trackH);
  
  Handle<reco::TrackCollection> trackOrigH;
  iEvent.getByToken(inputTrackOrig_, trackOrigH);
  
//   Handle<std::vector<int> > indicesH;
//   iEvent.getByToken(inputIndices_, indicesH);
  
//   Handle<std::vector<Trajectory> > trajH;
//   iEvent.getByToken(inputTraj_, trajH);
  
  Handle<reco::BeamSpot> bsH;
  iEvent.getByToken(inputBs_, bsH);
  
  const reco::BeamSpot& bs = *bsH;

//   const float mass = 0.105;
//   const float maxDPhi = 1.6;
//   PropagatorWithMaterial rPropagator(oppositeToMomentum, mass, field, maxDPhi, true, -1., false);
//   PropagatorWithMaterial fPropagator(alongMomentum, mass, field, maxDPhi, true, -1., false);
  
  std::unique_ptr<PropagatorWithMaterial> fPropagator(static_cast<PropagatorWithMaterial*>(thePropagator->clone()));
  fPropagator->setPropagationDirection(alongMomentum);
  
  KFUpdator updator;
  TkClonerImpl const& cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(ttrh.product())->cloner();

  
//   TkClonerImpl hitCloner;
//   TKCloner const* cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->cloner()
//   TrajectoryStateCombiner combiner;
  
  run = iEvent.run();
  lumi = iEvent.luminosityBlock();
  event = iEvent.id().event();
  
  for (const reco::Track &track : *trackOrigH) {
//     const Trajectory& traj = (*trajH)[itraj];
    
//     const edm::Ref<std::vector<Trajectory> > trajref(trajH, j);
//     const reco::Track& track = *(*trackH)[trajref];
//     const reco::Track& track = (*trackH)[itraj];
//     const reco::Track& trackOrig = (*trackOrigH)[(*indicesH)[j]];

//     std::cout << "j " << j << " (*indicesH)[j] " << (*indicesH)[j] <<std::endl;
    
    if (track.isLooper()) {
      continue;
    }
    trackPt = track.pt();
    trackEta = track.eta();
    trackPhi = track.phi();
    trackCharge = track.charge();
    trackPtErr = track.ptError();
    
    normalizedChi2 = track.normalizedChi2();
    
//     std::cout << "track pt: " << trackPt << " track eta: " << trackEta << " trackCharge: " << trackCharge << " qop: " << track.parameters()[0] << std::endl;
    
    auto const& tkparms = track.parameters();
    auto const& tkcov = track.covariance();
    trackParms.fill(0.);
    trackCov.fill(0.);
    //use eigen to fill raw memory
    Map<Vector5f>(trackParms.data()) = Map<const Vector5d>(tkparms.Array()).cast<float>();
    Map<Matrix<float, 5, 5, RowMajor> >(trackCov.data()).triangularView<Upper>() = Map<const Matrix<double, 5, 5, RowMajor> >(tkcov.Array()).cast<float>().triangularView<Upper>();
    
//     std::cout << "track charge: " << track.charge() << " trackorig charge " << trackOrig.charge() << "inner state charge " << tms.back().updatedState().charge() << std::endl;
    
    const reco::GenParticle* genpart = nullptr;
    
    genPt = -99.;
    genEta = -99.;
    genPhi = -99.;
    genCharge = -99;
    genParms.fill(0.);
    for (std::vector<reco::GenParticle>::const_iterator g = genParticles.begin(); g != genParticles.end(); ++g)
    {
      if (g->status() != 1) {
        continue;
      }
      if (std::abs(g->pdgId()) != 13) {
        continue;
      }
      
      float dR = deltaR(g->phi(), trackPhi, g->eta(), trackEta);
      
      if (dR < 0.15)
      {
        genpart = &(*g);
        
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
    

//     PropagationDirection rpropdir = traj.direction();
//     PropagationDirection fpropdir = rpropdir == alongMomentum ? oppositeToMomentum : alongMomentum;
    
    //TODO properly handle the outside-in case
    assert(track.seedDirection() == alongMomentum);
    
    const unsigned int nhits = track.recHitsSize();
    nHits = nhits;
//     unsigned int npixhits = 0;

    unsigned int nvalid = 0;
    unsigned int nvalidpixel = 0;
    
    //count valid hits since this is needed to size the arrays
    auto const& hitsbegin = track.recHitsBegin();
    for (unsigned int ihit = 0; ihit < track.recHitsSize(); ++ihit) {
      auto const& hit = *(hitsbegin + ihit);
      if (hit->isValid() && hit->dimension()<=2) {
        nvalid += 1;
        const GeomDet *detectorG = globalGeometry->idToDet(hit->geographicalId());
        if (hit->dimension()==2 && GeomDetEnumerators::isTrackerPixel(detectorG->subDetector())) {
          nvalidpixel += 1;
        }
      }
    }
    
    nValidHits = nvalid;
    nValidPixelHits = nvalidpixel;
    
//     const unsigned int nstriphits = nhits-npixhits;
//     const unsigned int nparsAlignment = nstriphits + 2*npixhits;
    const unsigned int nvalidstrip = nvalid - nvalidpixel;
    const unsigned int nparsAlignment = nvalidstrip + 2*nvalidpixel;
    const unsigned int nparsBfield = nhits;
    const unsigned int nparsEloss = nhits-1;
    const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
    
//     const unsigned int nstateparms = 5*(nhits+1);
//     const unsigned int nstateparms = 3*(nhits+1) - 1;
    const unsigned int nstateparms = 3*nhits - 1;
    const unsigned int nparmsfull = nstateparms + npars;
    
    
//     std::cout << "nhits " << nhits << std::endl;
//     std::cout << "nstateparms " << nstateparms << std::endl;
//     std::cout << "nparmsfull " << nparmsfull << std::endl;
//     std::cout << "nparmsfull " << nparmsfull << std::endl;
//     std::cout << "nparmsfull " << nparmsfull << std::endl;
//     std::cout << "nparmsfull " << nparmsfull << std::endl;
//     std::cout << "nparmsfull " << nparmsfull << std::endl;
    
//     const unsigned int npropparms = 5*(nhits-1);
//     const unsigned int nhitparms = 2*nhits;
//     const unsigned int nmomparms = 3*(nhits-1);
//     const unsigned int nposparms = 2*(nhits-1);
//     constexpr unsigned int nrefparms = 5;
    

    
    //active double for autodiff gradients
//     using Adouble = AutoDiffScalar<VectorXd>;
//     using AVectorXd = Matrix<Adouble, Dynamic, 1>;
//     //double double for autodiff hessians
//     using AAdouble = AutoDiffScalar<AVectorXd>;
    

    
//     using AAXd = AANT<double, Dynamic>;
//     using AAdouble = AAXd;
//     
//     using AA2d = AANT<double, 2>;
//     using AA3d = AANT<double, 3>;
//     using AA4d = AANT<double, 4>;
//     using AA12d = AANT<double, 12>;
//     
//     using ScalarConst = AANT<double, 0>;
    
//     using AConstd = AutoDiffScalar<VectorXd>;
//     using AConstd = AutoDiffScalar<Matrix<double, 0, 0>>;
    
    
//     using VectorXAd = Matrix<AScalar, Dynamic, 1>;
//     using MatrixXAd = Matrix<AScalar, Dynamic, Dynamic>;
    
    //two position parameters and and one alignment parameter
    using StripHitScalar = AANT<double, 3>;;
    
    using StripHit1DJacobian = Matrix<StripHitScalar, 1, 2>;
    
    using StripHitVector = Matrix<StripHitScalar, 2, 1>;
    using StripHit2DCovariance = Matrix<StripHitScalar, 2, 2>;
    using StripHit2DJacobian = Matrix<StripHitScalar, 2, 2>;

    
    
    //two hit dimensions and two alignment parameters
    using PixelHit2DScalar = AANT<double, 4>;
    using PixelHit2DVector = Matrix<PixelHit2DScalar, 2, 1>;
    using PixelHit2DCovariance = Matrix<PixelHit2DScalar, 2, 2>;
    using PixelHit2DJacobian = Matrix<PixelHit2DScalar, 2, 2>;
    
    
    //2x5 state parameters, one bfield parameter, and one material parameter
    using MSScalar = AANT<double, 11>;;
    using MSVector = Matrix<MSScalar, 5, 1>;
    using MSProjection = Matrix<MSScalar, 5, 5>;
    using MSJacobian = Matrix<MSScalar, 5, 5>;
    using MSCovariance = Matrix<MSScalar, 5, 5>;

//     using HitProjection = Matrix<AAdouble, 2, 5>;
//     using HitCovariance = Matrix<AAdouble, 2, 2>;
//     using HitVector = Matrix<AAdouble, 2, 1>;
    
//     evector<HitCovarianceMatrix> Vinv(nhits, HitCovarianceMatrix::Zero());
//     evector<HitProjection> Hh(nhits, HitProjection::Zero());
//     evector<HitVector> dy0(nhits, HitVector::Zero());
//     evector<StateVector> dx(nhits, StateVector::Zero());
//     //initialize backpropagation indexing
//     for (unsigned int i=0; i<nhits; ++i) {
// //       StateVector& dxi = dx[i];
// //       dxi.derivatives().resize(nstateparms);
//       for (unsigned int j=0; j<5; ++j) {
// //         dx[i][j].derivatives() = VectorXd::Unit(nstateparms, 5*i + j);
//         init_twice_active_var(dx[i][j], nstateparms, 5*i +j);
//       }
//     }
    
    
    VectorXd gradfull;
    MatrixXd hessfull;
    
    
//     VectorXd gradfull = chisq.value().derivatives();
//     MatrixXd hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);
//     for (unsigned int i=0; i<nstateparms; ++i) {
//       hessfull.row(i) = chisq.derivatives()[i].derivatives();
//     }
    
    
    globalidxv.clear();
    globalidxv.resize(npars, 0);
    
    nParms = npars;
    if (fillTrackTree_) {
      tree->SetBranchAddress("globalidxv", globalidxv.data());
    }
    
//     TrajectoryStateOnSurface currtsos;
    
    
    
    VectorXd dxfull;
    MatrixXd dxdparms;
    VectorXd grad;
    MatrixXd hess;
    LDLT<MatrixXd> Cinvd;
    
    if (dogen && genpart==nullptr) {
      continue;
    }
    
//     if (dogen && genpart->eta()>-2.3) {
//       continue;
//     }

//     if (genpart==nullptr) {
//       continue;
//     }
//     if (genpart->pt()>10.) {
//       continue;
//     }
//     if (genpart->pt()<100.) {
//       continue;
//     }
//     if (genpart->eta()>-2.3) {
//       continue;
//     }
    
    std::cout << "initial reference point parameters:" << std::endl;
    std::cout << track.parameters() << std::endl;

    //prepare hits
    TransientTrackingRecHit::RecHitContainer hits;
    hits.reserve(track.recHitsSize());
    for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
      const GeomDet *detectorG = globalGeometry->idToDet((*it)->geographicalId());
      hits.push_back((*it)->cloneForFit(*detectorG));
    }
    
    
    FreeTrajectoryState refFts;
    FreeTrajectoryState currentFts;
    
    if (dogen) {
      //init from gen state
      auto const& refpoint = genpart->vertex();
      auto const& trackmom = genpart->momentum();
      const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
      const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z()); 
      const GlobalTrajectoryParameters refglobal(refpos, refmom, genpart->charge(), field);
      
      //zero uncertainty on generated parameters
      AlgebraicSymMatrix55 nullerr;
      const CurvilinearTrajectoryError referr(nullerr);
      
//       refFts = FreeTrajectoryState(refpos, refmom, genpart->charge(), field);
      refFts = FreeTrajectoryState(refglobal, referr);
    }
    else {
      //init from track state
      auto const& refpoint = track.referencePoint();
      auto const& trackmom = track.momentum();
      const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
      const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z()); 
      const GlobalTrajectoryParameters refglobal(refpos, refmom, track.charge(), field);
      const CurvilinearTrajectoryError referr(track.covariance());
//       refFts = FreeTrajectoryState(refpos, refmom, track.charge(), field);
      refFts = FreeTrajectoryState(refglobal, referr);
    }

    std::vector<TrajectoryStateOnSurface> layerStates;
    layerStates.reserve(nhits);
    
    //inflate errors
    refFts.rescaleError(100.);
    
    
    bool valid = true;
//     unsigned int ntotalhitdim = 0;
//     unsigned int alignmentidx = 0;
//     unsigned int bfieldidx = 0;
//     unsigned int elossidx = 0;
    
//     constexpr unsigned int niters = 4;
    constexpr unsigned int niters = 1;
    
    for (unsigned int iiter=0; iiter<niters; ++iiter) {
      std::cout<< "iter " << iiter << std::endl;
      
      gradfull = VectorXd::Zero(nparmsfull);
      hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);
      unsigned int parmidx = 0;
      unsigned int alignmentparmidx = 0;

      if (iiter > 0) {
        //update current state from reference point state (errors not needed beyond first iteration)
        JacobianCurvilinearToCartesian curv2cart(refFts.parameters());
        const AlgebraicMatrix65& jac = curv2cart.jacobian();
        const AlgebraicVector6 glob = refFts.parameters().vector();
        
        auto const& dxlocal = dxfull.head<5>();
        const Matrix<double, 6, 1> globupd = Map<const Matrix<double, 6, 1>>(glob.Array()) + Map<const Matrix<double, 6, 5, RowMajor>>(jac.Array())*dxlocal;
        
        const GlobalPoint pos(globupd[0], globupd[1], globupd[2]);
        const GlobalVector mom(globupd[3], globupd[4], globupd[5]);
        double charge = std::copysign(1., refFts.charge()/refFts.momentum().mag() + dxlocal[0]);
//         std::cout << "before update: reffts:" << std::endl;
//         std::cout << refFts.parameters().vector() << std::endl;
//         std::cout << "charge " << refFts.charge() << std::endl;
        refFts = FreeTrajectoryState(pos, mom, charge, field);
//         std::cout << "after update: reffts:" << std::endl;
//         std::cout << refFts.parameters().vector() << std::endl;
//         std::cout << "charge " << refFts.charge() << std::endl;
        currentFts = refFts;
      }
      
      Matrix5d Hlm = Matrix5d::Identity();
      currentFts = refFts;
      
      auto propresult = fPropagator->geometricalPropagator().propagateWithPath(currentFts, *hits[0]->surface());
      if (!propresult.first.isValid()) {
        std::cout << "Abort: Propagation of reference state Failed!" << std::endl;
        valid = false;
        break;
      }
      
      for (unsigned int ihit = 0; ihit < hits.size(); ++ihit) {
//         std::cout << "ihit " << ihit << std::endl;
        
//         auto const& inhit = *(hitsbegin + ihit);
        //TODO check for null geographicalId?
//         const GeomDet *detectorG = globalGeometry->idToDet(inhit->geographicalId());
//         auto const& hit = inhit->cloneForFit(*detectorG);
  //       const TransientTrackingRecHit hit = *inhit->cloneForFit(*detectorG);
        
        auto const& hit = hits[ihit];
        
        TrajectoryStateOnSurface& updtsos = propresult.first;
        double sm = propresult.second;
        
        // inverse jacobian for propagated state (before material effects)
        JacobianLocalToCurvilinear hcm(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
        const AlgebraicMatrix55 &jachcm = hcm.jacobian();
        //efficient assignment from SMatrix using Eigen::Map
        const Map<const Matrix<double, 5, 5, RowMajor>> Hcm(jachcm.Array());
//         MSJacobian Hm = jachpropeig.cast<MSScalar>();
        
        //compute inverse transport jacobian
        // TODO use intended Bfield value (at propagation source)
//         AnalyticalCurvilinearJacobian curvjac(updtsos.globalParameters(), currentFts.parameters().position(), currentFts.parameters().momentum(), -sm);
        AnalyticalCurvilinearJacobian curvjac(currentFts.parameters(), updtsos.globalParameters().position(), updtsos.globalParameters().momentum(), sm);
        const AlgebraicMatrix55 &jacFm = curvjac.jacobian();
//         const Map<const Matrix<double, 5, 5, RowMajor>> Fm(jacFm.Array());
        const Matrix5d Fm = Map<const Matrix<double, 5, 5, RowMajor>>(jacFm.Array()).inverse();
        
//         MSJacobian Fm = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array()).cast<MSScalar>();
//         const Matrix<double, 5, 1> Fqop = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array()).col(0);
//         std::cout << "Fqop from CMSSW" << std::endl;
//         std::cout << Fqop << std::endl;
        
        // inverse bfield jacobian
        const Vector5d dFm = bfieldJacobian(updtsos.globalParameters(), currentFts.parameters(), -sm);
//         const MSVector dF = bfieldJacobian(currentFts.parameters(), updtsos.globalParameters(), s).cast<MSScalar>();
        
        //energy loss jacobian
        const Matrix<double, 5, 6> EdE = materialEffectsJacobian(updtsos, fPropagator->materialEffectsUpdator());
//         const MSJacobian E = EdE.leftCols<5>().cast<MSScalar>();
//         const MSVector dE = EdE.rightCols<1>().cast<MSScalar>();
        
        //process noise jacobians
//         const std::array<Matrix<double, 5, 5>, 5> dQs = processNoiseJacobians(updtsos, fPropagator->materialEffectsUpdator());
        
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
        
        ok = fPropagator->materialEffectsUpdator().updateStateInPlace(updtsos, alongMomentum);
        if (!ok) {
          std::cout << "Abort: material update failed" << std::endl;
          valid = false;
          break;
        }
        
        //get the process noise matrix
        AlgebraicMatrix55 const Qmat = tmptsos.localError().matrix();
        const Map<const Matrix<double, 5, 5, RowMajor>>Q(Qmat.Array());
        std::cout<< "Q" << std::endl;
        std::cout<< Q << std::endl;
//         MSCovariance Qinv = MSCovariance::Zero();
        //Q is 3x3 in the upper left block because there is no displacement on thin scattering layers
        //so invert the upper 3x3 block
//         Qinv.topLeftCorner<3,3>() = iQ.topLeftCorner<3,3>().inverse().cast<MSScalar>();
        
        //zero displacement on thin scattering layer approximated with small uncertainty
//         const double epsxy = 1e-5; //0.1um
//         Qinv(3,3) = MSScalar(1./epsxy/epsxy);
//         Qinv(4,4) = MSScalar(1./epsxy/epsxy);
        
        //apply measurement update if applicable
        auto const& preciseHit = hit->isValid() ? cloner.makeShared(hit, updtsos) : hit;
        if (hit->isValid() && !preciseHit->isValid()) {
          std::cout << "Abort: Failed updating hit" << std::endl;
          valid = false;
          break;
        }
        //momentum kink residual
        //TODO rework code so this can be computed together with the update without loss of precision
  //       const AlgebraicVector5 idx0 = hit->isValid() ? update(updtsos, *preciseHit) : AlgebraicVector5(0., 0., 0., 0., 0.);
  //       updtsos = hit->isValid() ? updator.update(updtsos, *preciseHit) : updtsos;

        AlgebraicVector5 idx0(0., 0., 0., 0., 0.);
        if (iiter==0) {
          //current state from predicted state
//           if (hit->isValid()) {
          if (false) {
            idx0 = update(updtsos, *preciseHit);
//             std::cout << "before KF update, ihit " << ihit << std::endl;
//             std::cout << updtsos.localParameters().vector() << std::endl;
            updtsos = updator.update(updtsos, *preciseHit);
            if (!updtsos.isValid()) {
              std::cout << "Abort: Kalman filter update failed" << std::endl;
              valid = false;
              break;
            }
//             std::cout << "after KF update" << std::endl;
//             std::cout << updtsos.localParameters().vector() << std::endl;
          }
          layerStates.push_back(updtsos);
        }
        else {
          //current state from previous state on this layer
          //save current parameters          
          TrajectoryStateOnSurface& oldtsos = layerStates[ihit];
          JacobianCurvilinearToLocal curv2local(oldtsos.surface(), oldtsos.localParameters(), *oldtsos.magneticField());
          const AlgebraicMatrix55& jac = curv2local.jacobian();
          const AlgebraicVector5 local = oldtsos.localParameters().vector();
          auto const& dxlocal = dxfull.segment<5>(5*(ihit+1));
          const Matrix<double, 5, 1> localupd = Map<const Matrix<double, 5, 1>>(local.Array()) + Map<const Matrix<double, 5, 5, RowMajor>>(jac.Array())*dxlocal;
          AlgebraicVector5 localvecupd(localupd[0],localupd[1],localupd[2],localupd[3],localupd[4]);
          
          idx0 = localvecupd - updtsos.localParameters().vector();
          
          const LocalTrajectoryParameters localparms(localvecupd, oldtsos.localParameters().pzSign());
          
//           std::cout << "before update: oldtsos:" << std::endl;
//           std::cout << oldtsos.localParameters().vector() << std::endl;
          oldtsos.update(localparms, oldtsos.surface(), field, oldtsos.surfaceSide());
//           std::cout << "after update: oldtsos:" << std::endl;
//           std::cout << oldtsos.localParameters().vector() << std::endl;
          updtsos = oldtsos;
        }
        
        currentFts = *updtsos.freeState();
        
        Vector5d dx0 = Map<const Vector5d>(idx0.Array());
//         MSVector dx0 = Map<const Vector5d>(idx0.Array()).cast<MSScalar>();
        
        //jacobian for updated state
//         JacobianCurvilinearToLocal h(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
//         const AlgebraicMatrix55 &jach = h.jacobian();
        //efficient assignment from SMatrix using Eigen::Map
//         Map<const Matrix<double, 5, 5, RowMajor> > jacheig(jach.Array());
  //       Hh.block<2,5>(2*i, 5*i) = jacheig.bottomRows<2>();
  //       StateJacobian H = jacheig.cast<AAdouble>();
        
        if (ihit < (nhits-1)) {
          // inverse jacobian for state to be propagated forward
          JacobianLocalToCurvilinear hcp(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
          const AlgebraicMatrix55& jachcp = hcp.jacobian();
          const Map<const Matrix<double, 5, 5, RowMajor>> Hcp(jachcp.Array());
          
          propresult = fPropagator->geometricalPropagator().propagateWithPath(updtsos, *hits[ihit+1]->surface());
          if (!propresult.first.isValid()) {
            std::cout << "Abort: Propagation Failed!" << std::endl;
            valid = false;
            break;
          }
          if (ihit>0) {

            
            const TrajectoryStateOnSurface& proptsos = propresult.first;
            double sp = propresult.second;
            

            
            // jacobian for forward propagated state
            JacobianCurvilinearToLocal hlp(proptsos.surface(), proptsos.localParameters(), *proptsos.magneticField());
            const AlgebraicMatrix55& jachlp = hlp.jacobian();
            const Map<const Matrix<double, 5, 5, RowMajor>> Hlp(jachlp.Array());

            // transport jacobian for forward propagation
            AnalyticalCurvilinearJacobian curvjac(updtsos.globalParameters(), proptsos.globalParameters().position(), proptsos.globalParameters().momentum(), sp);
            const AlgebraicMatrix55 &jacFp = curvjac.jacobian();
            const Map<const Matrix<double, 5, 5, RowMajor>> Fp(jacFp.Array());
            
            // bfield jacobian for forward propagation
            const Vector5d dFp = bfieldJacobian(updtsos.globalParameters(), proptsos.globalParameters(), sp);
                      
            constexpr unsigned int nlocalstate = 8;
            constexpr unsigned int nlocalbfield = 2;
            constexpr unsigned int nlocaleloss = 1;
            constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
            
            constexpr unsigned int nlocal = nlocalstate + nlocalbfield + nlocaleloss;
            
            constexpr unsigned int localstateidx = 0;
  //           constexpr unsigned int localbfieldidx = localstateidx + nlocalstate;
  //           constexpr unsigned int localelossidx = localbfieldidx + nlocalbfield;
            constexpr unsigned int localparmidx = localstateidx + nlocalstate;
            
  //           const unsigned int fullstateidx = 3*ihit;
            const unsigned int fullstateidx = 3*(ihit-1);
            const unsigned int fullparmidx = nstateparms + parmidx;
            
            // composed jacobian for backwards propagation
            const Matrix5d Flm = Hlm*Fm*Hcm;
            // composed jacobian for forwards propagation
            const Matrix5d Flp = Hlp*Fp*Hcp;
            
            // individual pieces, now starting to cast to active scalars for autograd,
            // as in eq (3) of https://doi.org/10.1016/j.cpc.2011.03.017
            // du/dum
            Matrix<MSScalar, 2, 2> Jm = Flm.block<2, 2>(3, 3).cast<MSScalar>();
            // (du/dalpham)^-1
            Matrix<MSScalar, 2, 2> Sinvm = Flm.block<2, 2>(3, 1).inverse().cast<MSScalar>();
            // du/dqopm
            Matrix<MSScalar, 2, 1> Dm = Flm.block<2, 1>(3, 0).cast<MSScalar>();
            // du/dBm
            Matrix<MSScalar, 2, 1> Bm = dFm.segment<2>(3).cast<MSScalar>();

            // du/dup
            Matrix<MSScalar, 2, 2> Jp = Flp.block<2, 2>(3, 3).cast<MSScalar>();
            // (du/dalphap)^-1
            Matrix<MSScalar, 2, 2> Sinvp = Flp.block<2, 2>(3, 1).inverse().cast<MSScalar>();
            // du/dqopp
            Matrix<MSScalar, 2, 1> Dp = Flp.block<2, 1>(3, 0).cast<MSScalar>();
            // du/dBp
            Matrix<MSScalar, 2, 1> Bp = dFp.segment<2>(3).cast<MSScalar>();
            
            std::cout << "Jm" << std::endl;
            std::cout << Jm << std::endl;
            std::cout << "Sinvm" << std::endl;
            std::cout << Sinvm << std::endl;
            std::cout << "Dm" << std::endl;
            std::cout << Dm << std::endl;
            std::cout << "Bm" << std::endl;
            std::cout << Bm << std::endl;
            
            std::cout << "Jp" << std::endl;
            std::cout << Jp << std::endl;
            std::cout << "Sinvp" << std::endl;
            std::cout << Sinvp << std::endl;
            std::cout << "Dp" << std::endl;
            std::cout << Dp << std::endl;
            std::cout << "Bp" << std::endl;
            std::cout << Bp << std::endl;
            
            // energy loss jacobians
  //           const MSJacobian E = EdE.leftCols<5>().cast<MSScalar>();
  //           const MSVector dE = EdE.rightCols<1>().cast<MSScalar>();
            
            const MSScalar Eqop(EdE(0,0));
            const Matrix<MSScalar, 1, 2> Ealpha = EdE.block<1, 2>(0, 1).cast<MSScalar>();
            const MSScalar dE(EdE(0,5));
            
            //energy loss inverse variance
            MSScalar invSigmaE(1./Q(0,0));
            
            // multiple scattering inverse covariance
            Matrix<MSScalar, 2, 2> Qinvms = Q.block<2,2>(1,1).inverse().cast<MSScalar>();
            
            // initialize active scalars for state parameters
            Matrix<MSScalar, 2, 1> dum = Matrix<MSScalar, 2, 1>::Zero();
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
//             MSScalar dbeta(0.);
//             init_twice_active_var(dbeta, nlocal, localparmidx);
            
//             MSScalar dxi(0.);
//             init_twice_active_var(dxi, nlocal, localparmidx + 1);
            
//             MSScalar dbetap(0.);
//             init_twice_active_var(dbetap, nlocal, localparmidx + 2);
            
            //multiple scattering kink term
            
            const Matrix<MSScalar, 2, 1> dalpha0 = dx0.segment<2>(1).cast<MSScalar>();
            
//             const Matrix<MSScalar, 2, 1> dalpham = Sinvm*(dum - Jm*du - Dm*dqopm - Bm*dbeta);
//             const Matrix<MSScalar, 2, 1> dalphap = Sinvp*(dup - Jp*du - Dp*dqop - Bp*dbetap);
            const Matrix<MSScalar, 2, 1> dalpham = Sinvm*(dum - Jm*du - Dm*dqopm);
            const Matrix<MSScalar, 2, 1> dalphap = Sinvp*(dup - Jp*du - Dp*dqop);
            
            const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
            const MSScalar chisqms = dms.transpose()*Qinvms*dms;
//             (void)chisqms;
            
            //energy loss term
            const MSScalar deloss0(dx0[0]);
            
//             const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
            const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0];
            const MSScalar chisqeloss = deloss*deloss*invSigmaE;
//             (void)chisqeloss;
            
            const MSScalar chisq = chisqms + chisqeloss;
//             const MSScalar chisq = chisqms;

  //           std::cout << "chisq.value()" << std::endl;
  //           std::cout << chisq.value() << std::endl;
  //           std::cout << "chisq.value().derivatives()" << std::endl;
  //           std::cout << chisq.value().derivatives() << std::endl;
  //           std::cout << "chisq.derivatives()[0].derivatives()" << std::endl;
  //           std::cout << chisq.derivatives()[0].derivatives() << std::endl;
            
            
            //           const MSVector dms = dx0 + H*dx - E*Hprop*F*dxprev - E*Hprop*dF*dbeta - dE*dxi;
            
            
            
            
  //           MSScalar chisq;
  //           
  //           if (ihit==0 || ihit == (nhits-1)) {
  //             //standard fit
  //             const MSVector dms = dx0 + H*dx - E*Hprop*F*dxprev - E*Hprop*dF*dbeta - dE*dxi;
  //             chisq = dms.transpose()*Qinv*dms;            
  //           }
  //           else {
  //             //maximum likelihood fit
  //             const MSVector dxprop = Hprop*F*dxprev;
  //             const MSCovariance dQdxprop0 = dQs[0].cast<MSScalar>();
  //             const MSCovariance dQdxprop1 = dQs[1].cast<MSScalar>();
  //             const MSCovariance dQdxprop2 = dQs[2].cast<MSScalar>();
  //             const MSCovariance dQdxi = dQs[3].cast<MSScalar>();
  //             
  //             const MSCovariance dQ = dxprop[0]*dQdxprop0 + dxprop[1]*dQdxprop1 + dxprop[2]*dQdxprop2 + dxi*dQdxi;
  //             
  // //             const MSCovariance dQdxprop0 = dQs[0].cast<MSScalar>();
  // //             const MSCovariance d2Qdxprop02 = dQs[1].cast<MSScalar>();
  // //   //         
  // //             const MSCovariance dQ = dxprop[0]*dQdxprop0 + 0.5*dxprop[0]*dxprop[0]*d2Qdxprop02;
  //             
  // //             const Matrix<MSScalar, 3, 3> Qms = iQ.topLeftCorner<3,3>().cast<MSScalar>() + dQ.topLeftCorner<3,3>();
  // //             Qinv.topLeftCorner<3,3>() = Qms.inverse();
  //             const Matrix<MSScalar, 2, 2> Qms = iQ.block<2,2>(1,1).cast<MSScalar>() + dQ.block<2,2>(1,1);
  //             Qinv.block<2,2>(1,1) = Qms.inverse();
  //             
  //             const MSScalar logdetQ = Eigen::log(Qms.determinant());
  // 
  //             const MSVector dms = dx0 + H*dx - E*dxprop - E*Hprop*dF*dbeta - dE*dxi;
  //             chisq = dms.transpose()*Qinv*dms;
  //             chisq = chisq + logdetQ;            
  //             
  //           }
            
    //         MSCovariance Q = iQ.cast<MSScalar>();
            
    //         const MSVector dxprop = Hprop*F*dxprev;
    //         const MSCovariance dQdxprop0 = dQs[0].cast<MSScalar>();
    //         const MSCovariance dQdxprop1 = dQs[1].cast<MSScalar>();
    //         const MSCovariance dQdxprop2 = dQs[2].cast<MSScalar>();
    //         const MSCovariance dQdxi = dQs[3].cast<MSScalar>();
    // //         
    //         const MSCovariance dQ = dxprop[0]*dQdxprop0 + dxprop[1]*dQdxprop1 + dxprop[2]*dQdxprop2 + dxi*dQdxi;
      
    //         const MSVector dxprop = Hprop*F*dxprev;
    //         const MSCovariance dQdxprop0 = dQs[0].cast<MSScalar>();
    //         const MSCovariance d2Qdxprop02 = dQs[1].cast<MSScalar>();
    //         
    //         const MSCovariance dQ = dxprop[0]*dQdxprop0 + 0.5*dxprop[0]*dxprop[0]*d2Qdxprop02;
            
    //         MSCovariance Qinv = MSCovariance::Zero();
    //         Qinv(3,3) = MSScalar(1./epsxy/epsxy);
    //         Qinv(4,4) = MSScalar(1./epsxy/epsxy);
    //         Qinv.block<2,2>(1,1) = iQ.block<2,2>(1,1).inverse().cast<MSScalar>();
    //         const MSScalar Qelos = MSScalar(iQ(0,0)) + dQ(0,0);
    //         Qinv(0,0) = 1./Qelos;
    // //         const Matrix<MSScalar, 3, 3> Qms = iQ.topLeftCorner<3,3>().cast<MSScalar>() + dQ.topLeftCorner<3,3>();
    // //         Qinv.topLeftCorner<3,3>() = Qms.inverse();
    // //         const MSScalar logdetQ = Eigen::log(Qms.determinant());
    //         const MSScalar logdetQ = Eigen::log(Qelos);
    // //         
    //         const MSVector dms = dx0 + H*dx - E*dxprop - E*Hprop*dF*dbeta - dE*dxi;
    //         MSScalar chisq = dms.transpose()*Qinv*dms;
    //         chisq = chisq + logdetQ;
            
    //         const MSCovariance Qinvmod = Qinv - Qinv*dQ*Qinv;
    //         const MSScalar dlogdetQ = Eigen::log(1. + (Qinv*dQ).trace());
    //         
    //         const MSVector dms = dx0 + H*dx - E*dxprop - E*Hprop*dF*dbeta - dE*dxi;
    //         MSScalar chisq = dms.transpose()*Qinvmod*dms;
    //         chisq = chisq + dlogdetQ;

            
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
            
            const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(2,hit->geographicalId()));
            globalidxv[parmidx] = bfieldglobalidx;
            parmidx++;
            
            const unsigned int elossglobalidx = detidparms.at(std::make_pair(3,hit->geographicalId()));
            globalidxv[parmidx] = elossglobalidx;
            parmidx++;
          }
          
          
          // jacobian for state to be propagated forward (needed for next layer)
          JacobianCurvilinearToLocal hlm(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
          const AlgebraicMatrix55& jachlm = hlm.jacobian();
          Hlm = Map<const Matrix<double, 5, 5, RowMajor>>(jachlm.Array());
          
        }
        
        //hit information
        //FIXME consolidate this special cases into templated function(s)
        if (preciseHit->isValid()) {
          constexpr unsigned int nlocalstate = 2;
          constexpr unsigned int localstateidx = 0;
          constexpr unsigned int localalignmentidx = nlocalstate;
          constexpr unsigned int localparmidx = localalignmentidx;
          
//           const unsigned int fullstateidx = 3*(ihit+1);
          const unsigned int fullstateidx = 3*ihit;
          const unsigned int fullparmidx = nstateparms + nparsBfield + nparsEloss + alignmentparmidx;

          if (preciseHit->dimension()==1) {
            constexpr unsigned int nlocalalignment = 1;
            constexpr unsigned int nlocalparms = nlocalalignment;
            constexpr unsigned int nlocal = nlocalstate + nlocalparms;
            
            const StripHitScalar dy0(preciseHit->localPosition().x() - updtsos.localPosition().x());
            const StripHitScalar Vinv(1./preciseHit->localPositionError().xx());
            
            StripHitVector dx = StripHitVector::Zero();
            for (unsigned int j=0; j<dx.size(); ++j) {
              init_twice_active_var(dx[j], nlocal, localstateidx + j);
            }
            
            //single alignment parameter
            StripHitScalar dalpha(0.);
            init_twice_active_var(dalpha, nlocal, localalignmentidx);

            
            const StripHitScalar A(1.);
            
//             const StripHit1DJacobian H = jacheig.block<1,2>(3,3).cast<StripHitScalar>();
            
            StripHitScalar dh = dy0 - dx[0] - A*dalpha;
            StripHitScalar chisq = dh*dh*Vinv;
            
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
            
            const unsigned int xglobalidx = detidparms.at(std::make_pair(0,preciseHit->geographicalId()));
            globalidxv[nparsBfield + nparsEloss + alignmentparmidx] = xglobalidx;
            alignmentparmidx++;
          }
          else if (preciseHit->dimension()==2) {
            bool ispixel = GeomDetEnumerators::isTrackerPixel(preciseHit->det()->subDetector());

            Matrix2d iV;
            iV << preciseHit->localPositionError().xx(), preciseHit->localPositionError().xy(),
                  preciseHit->localPositionError().xy(), preciseHit->localPositionError().yy();
            
            if (ispixel) {
              constexpr unsigned int nlocalalignment = 2;
              constexpr unsigned int nlocalparms = nlocalalignment;
              constexpr unsigned int nlocal = nlocalstate + nlocalparms;
              
              PixelHit2DVector dy0;
              dy0[0] = PixelHit2DScalar(preciseHit->localPosition().x() - updtsos.localPosition().x());
              dy0[1] = PixelHit2DScalar(preciseHit->localPosition().y() - updtsos.localPosition().y());
              
              const PixelHit2DCovariance Vinv = iV.inverse().cast<PixelHit2DScalar>();
              
              PixelHit2DVector dx = PixelHit2DVector::Zero();
              for (unsigned int j=0; j<dx.size(); ++j) {
                init_twice_active_var(dx[j], nlocal, localstateidx + j);
              }
              
              //two alignment parameters
              PixelHit2DVector dalpha = PixelHit2DVector::Zero();
              for (unsigned int idim=0; idim<2; ++idim) {
                init_twice_active_var(dalpha[idim], nlocal, localalignmentidx+idim);
              }
              const Matrix<PixelHit2DScalar, 2, 2> A = Matrix<PixelHit2DScalar, 2, 2>::Identity();
              
//               const PixelHit2DJacobian H = jacheig.bottomRightCorner<2,2>().cast<PixelHit2DScalar>();

              const PixelHit2DVector dh = dy0 - dx - A*dalpha;
              const PixelHit2DScalar chisq = dh.transpose()*Vinv*dh;
              
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
              
              for (unsigned int idim=0; idim<2; ++idim) {
                const unsigned int xglobalidx = detidparms.at(std::make_pair(idim, preciseHit->geographicalId()));
                globalidxv[nparsBfield + nparsEloss + alignmentparmidx] = xglobalidx;
                alignmentparmidx++;
              }
            }
            else {
              constexpr unsigned int nlocalalignment = 1;
              constexpr unsigned int nlocalparms = nlocalalignment;
              constexpr unsigned int nlocal = nlocalstate + nlocalparms;
              
              StripHitVector dy0;
              dy0[0] = StripHitScalar(preciseHit->localPosition().x() - updtsos.localPosition().x());
              dy0[1] = StripHitScalar(preciseHit->localPosition().y() - updtsos.localPosition().y());
              
              const StripHit2DCovariance Vinv = iV.inverse().cast<StripHitScalar>();
              
              StripHitVector dx = StripHitVector::Zero();
              for (unsigned int j=0; j<dx.size(); ++j) {
                init_twice_active_var(dx[j], nlocal, localstateidx + j);
              }
              
              StripHitScalar dalpha(0.);
              init_twice_active_var(dalpha, nlocal, localalignmentidx);

              Matrix<StripHitScalar, 2, 1> A = Matrix<StripHitScalar, 2, 1>::Zero();
              A(0,0) = StripHitScalar(1.);
              
//               const StripHit2DJacobian H = jacheig.bottomRightCorner<2,2>().cast<StripHitScalar>();

              StripHitVector dh = dy0 - dx - A*dalpha;
              StripHitScalar chisq = dh.transpose()*Vinv*dh;

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
              
              const unsigned int xglobalidx = detidparms.at(std::make_pair(0,preciseHit->geographicalId()));
              globalidxv[nparsBfield + nparsEloss + alignmentparmidx] = xglobalidx;
              alignmentparmidx++;
              
            }
          }
        }
      }
      
      if (!valid) {
        break;
      }
      
      //fake constraint on reference point parameters
      if (dogen) {
        for (unsigned int i=0; i<5; ++i) {
          hessfull(i,i) = 1e6;
        }
      }
      
      //now do the expensive calculations and fill outputs
      auto const& dchisqdx = gradfull.head(nstateparms);
      auto const& dchisqdparms = gradfull.tail(npars);
      
      auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
      auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
      auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
      
      std::cout << "dchisqdx" << std::endl;
      std::cout << dchisqdx << std::endl;
      std::cout << "d2chisqdx2 diagonal" << std::endl;
      std::cout << d2chisqdx2.diagonal() << std::endl;
      std::cout << "d2chisqdx2" << std::endl;
      std::cout << d2chisqdx2 << std::endl;
      
  //     auto const& eigenvalues = d2chisqdx2.eigenvalues();
  //     std::cout << "d2chisqdx2 eigenvalues" << std::endl;
  //     std::cout << eigenvalues << std::endl;
      
//       auto const& Cinvd = d2chisqdx2.ldlt();
      Cinvd.compute(d2chisqdx2);
      
      dxfull = -Cinvd.solve(dchisqdx);
      
//       const Vector5d dxRef = dx.head<5>();
// //       const Vector5d dxRef = -Cinvd.solve(dchisqdx).head<5>();
//       const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();
      
//       dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
//       
//       grad = dchisqdparms + dxdparms*dchisqdx;
//       hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
//       
      std::cout << "dxfull" << std::endl;
      std::cout << dxfull << std::endl;
      std::cout << "errsq" << std::endl;
      std::cout << Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).diagonal() << std::endl;
      
      const Vector5d dxRef = dxfull.head<5>();
      const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();

      std::cout<< "dxRef" << std::endl;
      std::cout<< dxRef << std::endl;
      
      //fill output with corrected state and covariance at reference point
      refParms.fill(0.);
      refCov.fill(0.);
//       const AlgebraicVector5& refVec = track.parameters();
      CurvilinearTrajectoryParameters curvparms(refFts.position(), refFts.momentum(), refFts.charge());
      const AlgebraicVector5& refVec = curvparms.vector();
      Map<Vector5f>(refParms.data()) = (Map<const Vector5d>(refVec.Array()) + dxRef).cast<float>();
      Map<Matrix<float, 5, 5, RowMajor> >(refCov.data()).triangularView<Upper>() = (2.*Cinner).cast<float>().triangularView<Upper>();
      
      if (iiter==0) {
        refParms_iter0 = refParms;
        refCov_iter0 = refCov;
      }
//       else if (iiter==2) {
//         refParms_iter2 = refParms;
//         refCov_iter2 = refCov;        
//       }
      
      std::cout << "refParms" << std::endl;
      std::cout << Map<const Vector5f>(refParms.data()) << std::endl;
      
//   //     gradv.clear();
//       jacrefv.clear();
// 
//   //     gradv.resize(npars,0.);
//       jacrefv.resize(5*npars, 0.);
//       
//       nJacRef = 5*npars;
//   //     tree->SetBranchAddress("gradv", gradv.data());
//       tree->SetBranchAddress("jacrefv", jacrefv.data());
//       
//       //eigen representation of the underlying vector storage
//   //     Map<VectorXf> gradout(gradv.data(), npars);
//       Map<Matrix<float, 5, Dynamic, RowMajor> > jacrefout(jacrefv.data(), 5, npars);
//       
//       jacrefout = dxdparms.leftCols<5>().transpose().cast<float>();
    
    }
    
    if (!valid) {
      continue;
    }
    
    auto const& dchisqdx = gradfull.head(nstateparms);
    auto const& dchisqdparms = gradfull.tail(npars);
    
    auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
    auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
    auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
    
    dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
    
    grad = dchisqdparms + dxdparms*dchisqdx;
    //TODO check the simplification
//     hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
    hess = d2chisqdparms2 + dxdparms*d2chisqdxdparms;
    
//     const Vector5d dxRef = dxfull.head<5>();
    const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();

    
//     gradv.clear();
    jacrefv.clear();

//     gradv.resize(npars,0.);
    jacrefv.resize(5*npars, 0.);
    
    nJacRef = 5*npars;
//     tree->SetBranchAddress("gradv", gradv.data());
    if (fillTrackTree_) {
      tree->SetBranchAddress("jacrefv", jacrefv.data());
    }
    
    //eigen representation of the underlying vector storage
//     Map<VectorXf> gradout(gradv.data(), npars);
    Map<Matrix<float, 5, Dynamic, RowMajor> > jacrefout(jacrefv.data(), 5, npars);
    
    jacrefout = dxdparms.leftCols<5>().transpose().cast<float>();    
    
//     gradout = grad.cast<float>();
    
    
    float refPt = std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1]);

    gradmax = 0.;
    for (unsigned int i=0; i<npars; ++i) {
      const float absval = std::abs(grad[i]);
      if (absval>gradmax) {
        gradmax = absval;
      }      
    }
    
    
    if (gradmax < 1e5 && refPt > 5.5) {
      //fill aggregrate gradient and hessian
      for (unsigned int i=0; i<npars; ++i) {
        gradagg[globalidxv[i]] += grad[i];
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
          
          const std::pair<unsigned int, unsigned int> key = std::make_pair(std::min(iidx,jidx), std::max(iidx,jidx));
          
          auto it = hessaggsparse.find(key);
          if (it==hessaggsparse.end()) {
            hessaggsparse[key] = hess(i,j);
          }
          else {
            it->second += hess(i,j);
          }
        }
      }
    }
    
    
    std::cout << "hess debug" << std::endl;
    std::cout << "track parms" << std::endl;
    std::cout << tkparms << std::endl;
//     std::cout << "dxRef" << std::endl;
//     std::cout << dxRef << std::endl;
    std::cout << "original cov" << std::endl;
    std::cout << track.covariance() << std::endl;
    std::cout << "recomputed cov" << std::endl;
    std::cout << 2.*Cinner << std::endl;

//     std::cout << "dxinner/dparms" << std::endl;
//     std::cout << dxdparms.bottomRows<5>() << std::endl;
//     std::cout << "grad" << std::endl;
//     std::cout << grad << std::endl;
//     std::cout << "hess diagonal" << std::endl;
//     std::cout << hess.diagonal() << std::endl;
//     std::cout << "hess0 diagonal" << std::endl;
//     std::cout << d2chisqdparms2.diagonal() << std::endl;
//     std::cout << "hess1 diagonal" << std::endl;
//     std::cout << 2.*(dxdparms.transpose()*d2chisqdxdparms).diagonal() << std::endl;
//     std::cout << "hess2 diagonal" << std::endl;
//     std::cout << (dxdparms.transpose()*d2chisqdx2*dxdparms).diagonal() << std::endl;
    
    //fill packed hessian and indices
//     const unsigned int nsym = npars*(1+npars)/2;
//     hesspackedv.clear();    
//     hesspackedv.resize(nsym, 0.);
//     
//     nSym = nsym;
//     tree->SetBranchAddress("hesspackedv", hesspackedv.data());
//     
//     Map<VectorXf> hesspacked(hesspackedv.data(), nsym);
//     const Map<const VectorXu> globalidx(globalidxv.data(), npars);
// 
//     unsigned int packedidx = 0;
//     for (unsigned int ipar = 0; ipar < npars; ++ipar) {
//       const unsigned int segmentsize = npars - ipar;
//       hesspacked.segment(packedidx, segmentsize) = hess.block<1, Dynamic>(ipar, ipar, 1, segmentsize).cast<float>();
//       packedidx += segmentsize;
//     }

    if (fillTrackTree_) {
      tree->Fill();
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ResidualGlobalCorrectionMaker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void ResidualGlobalCorrectionMaker::endJob()
{
//   fout->cd();
  
//   TTree *gradtree = new TTree("gradtree","");
  unsigned int idx;
  double gradval;
  gradtree->Branch("idx",&idx);
  gradtree->Branch("gradval",&gradval);
  for (unsigned int i=0; i<gradagg.size(); ++i) {
    idx = i;
    gradval = gradagg[i];
    gradtree->Fill();
  }
  
//   TTree *hesstree = new TTree("hesstree","");
  unsigned int iidx;
  unsigned int jidx;
  double hessval;
  hesstree->Branch("iidx",&iidx);
  hesstree->Branch("jidx",&jidx);
  hesstree->Branch("hessval",&hessval);
  
  for (auto const& item : hessaggsparse) {
    iidx = item.first.first;
    jidx = item.first.second;
    hessval = item.second;
    hesstree->Fill();
  }
  
//   fout->Write();
//   fout->Close();
}

// ------------ method called when starting to processes a run  ------------

void 
ResidualGlobalCorrectionMaker::beginRun(edm::Run const& run, edm::EventSetup const& es)
{
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  es.get<GlobalTrackingGeometryRecord>().get(globalGeometry);
  
  edm::ESHandle<TrackerTopology> trackerTopology;
  es.get<TrackerTopologyRcd>().get(trackerTopology);
  
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
  
//   TFile *runfout = new TFile("trackTreeGradsParmInfo.root", "RECREATE");
//   TTree *runtree = new TTree("tree", "tree");
  
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
  float xi;

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
  runtree->Branch("xi", &xi);
  
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
//       TIBDetId detid(det->geographicalId());
//       layer = detid.layer();
      layer = trackerTopology->tibLayer(det->geographicalId());
    }
    else if (det->subDetector() == GeomDetEnumerators::TOB)
    {
//       TOBDetId detid(det->geographicalId());
//       layer = detid.layer();
      layer = trackerTopology->tobLayer(det->geographicalId());
    }
    else if (det->subDetector() == GeomDetEnumerators::TID)
    {
      unsigned int side = trackerTopology->tidSide(detid);
      unsigned int wheel = trackerTopology->tidWheel(detid);
      layer = -1 * (side == 1) * wheel + (side == 2) * wheel;

    }
    else if (det->subDetector() == GeomDetEnumerators::TEC)
    {
//       TECDetId detid(det->geographicalId());
//       layer = -1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel();
      unsigned int side = trackerTopology->tecSide(detid);
      unsigned int wheel = trackerTopology->tecWheel(detid);
      layer = -1 * (side == 1) * wheel + (side == 2) * wheel;
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
    xi = det->surface().mediumProperties().xi();
    
    //fill map
    detidparms.emplace(key, globalidx);
    globalidx++;
    
    runtree->Fill();
  }
  
//   runfout->Write();
//   runfout->Close();
  
  unsigned int nglobal = detidparms.size();
//   std::sort(detidparms.begin(), detidparms.end());
  std::cout << "nglobalparms = " << detidparms.size() << std::endl;
  
  //initialize gradient
  if (!gradagg.size()) {
    gradagg.resize(nglobal, 0.);
  }
    
  
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

Matrix<double, 5, 1> ResidualGlobalCorrectionMaker::bfieldJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s) const {
 
  //analytic jacobian wrt magnitude of magnetic field
  //TODO should we parameterize with respect to z-component instead?
  //extending derivation from CMS NOTE 2006/001
  const Vector3d b(globalSource.magneticFieldInInverseGeV().x(),
                      globalSource.magneticFieldInInverseGeV().y(),
                      globalSource.magneticFieldInInverseGeV().z());
  const double magb = b.norm();
  const Vector3d h = b.normalized();

  const Vector3d p0(globalSource.momentum().x(),
                      globalSource.momentum().y(),
                      globalSource.momentum().z());
  const Vector3d p1(globalDest.momentum().x(),
                      globalDest.momentum().y(),
                      globalDest.momentum().z());
  const Vector3d M0(globalSource.position().x(),
                    globalSource.position().y(),
                    globalSource.position().z());
  const Vector3d T0 = p0.normalized();
  const Vector3d T = p1.normalized();
  const double p = p0.norm();
  const double q = globalSource.charge();
  const double qop = q/p;

  const Vector3d N0alpha = h.cross(T0);
  const double alpha = N0alpha.norm();
  const Vector3d N0 = N0alpha.normalized();
  const double gamma = h.transpose()*T;
  const Vector3d Z(0.,0.,1.);
  const Vector3d U = Z.cross(T).normalized();
  const Vector3d V = T.cross(U);

  //this is printed from sympy.printing.cxxcode together with sympy.cse for automatic substitution of common expressions
  auto const xf0 = h;
  auto const xf1 = xf0[2];
  auto const xf2 = magb*qop;
  auto const xf3 = s*xf2;
  auto const xf4 = std::cos(xf3);
  auto const xf5 = 1 - xf4;
  auto const xf6 = gamma*xf5;
  auto const xf7 = T0;
  auto const xf8 = xf7[2];
  auto const xf9 = xf4*xf8;
  auto const xf10 = N0;
  auto const xf11 = xf10[2];
  auto const xf12 = std::sin(xf3);
  auto const xf13 = alpha*xf12;
  auto const xf14 = xf11*xf13;
  auto const xf15 = -xf14 + xf9;
  auto const xf16 = xf1*xf6 + xf15;
  auto const xf17 = std::pow(qop, -2);
  auto const xf18 = xf0[0];
  auto const xf19 = xf7[0];
  auto const xf20 = xf19*xf4;
  auto const xf21 = xf10[0];
  auto const xf22 = xf13*xf21;
  auto const xf23 = xf20 - xf22;
  auto const xf24 = xf18*xf6 + xf23;
  auto const xf25 = xf0[1];
  auto const xf26 = xf7[1];
  auto const xf27 = xf26*xf4;
  auto const xf28 = xf10[1];
  auto const xf29 = xf13*xf28;
  auto const xf30 = xf27 - xf29;
  auto const xf31 = xf25*xf6 + xf30;
  auto const xf32 = xf17*std::pow(xf24, 2) + xf17*std::pow(xf31, 2);
  auto const xf33 = xf17/xf32;
  auto const xf34 = 1.0/(std::pow(xf16, 2)*xf33 + 1);
  auto const xf35 = qop*s;
  auto const xf36 = xf12*xf8;
  auto const xf37 = xf12*xf35;
  auto const xf38 = gamma*xf1;
  auto const xf39 = xf35*xf4;
  auto const xf40 = alpha*xf11;
  auto const xf41 = 1.0/std::fabs(qop);
  auto const xf42 = xf41/std::sqrt(xf32);
  auto const xf43 = xf19*xf37;
  auto const xf44 = alpha*xf21;
  auto const xf45 = xf39*xf44;
  auto const xf46 = gamma*xf37;
  auto const xf47 = xf18*xf46;
  auto const xf48 = (1.0/2.0)*xf17;
  auto const xf49 = xf24*xf48;
  auto const xf50 = xf26*xf37;
  auto const xf51 = alpha*xf28;
  auto const xf52 = xf39*xf51;
  auto const xf53 = xf25*xf46;
  auto const xf54 = xf31*xf48;
  auto const xf55 = xf16*xf41/std::pow(xf32, 3.0/2.0);
  auto const xf56 = 1.0/magb;
  auto const xf57 = s*xf56;
  auto const xf58 = xf4*xf57;
  auto const xf59 = 1.0/qop;
  auto const xf60 = xf59/std::pow(magb, 2);
  auto const xf61 = xf12*xf60;
  auto const xf62 = xf58 - xf61;
  auto const xf63 = -gamma*xf58 + gamma*xf61;
  auto const xf64 = xf60*(-xf12*xf3 + xf5);
  auto const xf65 = -xf16*(xf1*xf63 + xf40*xf64 + xf62*xf8) - xf24*(xf18*xf63 + xf19*xf62 + xf44*xf64) - xf31*(xf25*xf63 + xf26*xf62 + xf51*xf64);
  auto const xf66 = xf12*xf2;
  auto const xf67 = xf2*xf4;
  auto const xf68 = xf19*xf66;
  auto const xf69 = xf44*xf67;
  auto const xf70 = gamma*xf66;
  auto const xf71 = xf18*xf70;
  auto const xf72 = xf26*xf66;
  auto const xf73 = xf51*xf67;
  auto const xf74 = xf25*xf70;
  auto const xf75 = xf24*xf33;
  auto const xf76 = xf31*xf33;
  auto const xf77 = U;
  auto const xf78 = xf77[0];
  auto const xf79 = xf5*xf60;
  auto const xf80 = gamma*xf18;
  auto const xf81 = xf60*(xf12 - xf3);
  auto const xf82 = xf56*xf59;
  auto const xf83 = xf82*(-xf35 + xf39);
  auto const xf84 = -xf19*xf61 + xf20*xf57 - xf22*xf57 + xf44*xf79 + xf80*xf81 - xf80*xf83;
  auto const xf85 = xf77[1];
  auto const xf86 = gamma*xf25;
  auto const xf87 = -xf26*xf61 + xf27*xf57 - xf29*xf57 + xf51*xf79 + xf81*xf86 - xf83*xf86;
  auto const xf88 = xf77[2];
  auto const xf89 = -xf14*xf57 + xf38*xf81 - xf38*xf83 + xf40*xf79 + xf57*xf9 - xf61*xf8;
  auto const xf90 = xf82*(-xf2 + xf67);
  auto const xf91 = xf23 - xf80*xf90;
  auto const xf92 = xf30 - xf86*xf90;
  auto const xf93 = xf15 - xf38*xf90;
  auto const xf94 = V;
  auto const xf95 = xf94[0];
  auto const xf96 = xf94[1];
  auto const xf97 = xf94[2];
  auto const dlamdB = xf34*xf65*(xf42*(-xf2*xf36 + xf38*xf66 - xf40*xf67) + xf55*(-xf49*(-2*xf68 - 2*xf69 + 2*xf71) - xf54*(-2*xf72 - 2*xf73 + 2*xf74))) + xf34*(xf42*(-xf35*xf36 + xf37*xf38 - xf39*xf40) + xf55*(-xf49*(-2*xf43 - 2*xf45 + 2*xf47) - xf54*(-2*xf50 - 2*xf52 + 2*xf53)));
  auto const dphidB = xf65*(xf75*(-xf72 - xf73 + xf74) - xf76*(-xf68 - xf69 + xf71)) + xf75*(-xf50 - xf52 + xf53) - xf76*(-xf43 - xf45 + xf47);
  auto const dxtdB = xf65*(xf78*xf91 + xf85*xf92 + xf88*xf93) + xf78*xf84 + xf85*xf87 + xf88*xf89;
  auto const dytdB = xf65*(xf91*xf95 + xf92*xf96 + xf93*xf97) + xf84*xf95 + xf87*xf96 + xf89*xf97;

  Matrix<double, 5, 1> dF;
  dF[0] = 0.;
  dF[1] = dlamdB;
  dF[2] = dphidB;
  dF[3] = dxtdB;
  dF[4] = dytdB;
  
//   convert to tesla
  dF *= 2.99792458e-3;
  
//   Matrix<double, 5, 1> Fqop;
//   Fqop[0] = 1.;
//   Fqop[1] = dlamdqop;
//   Fqop[2] = dphidqop;
//   Fqop[3] = dxtdqop;
//   Fqop[4] = dytdqop;
// //   
//   std::cout << "Fqop from sympy:" << std::endl;
//   std::cout << Fqop << std::endl;
  
  return dF;
}

Matrix<double, 5, 6> ResidualGlobalCorrectionMaker::materialEffectsJacobian(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) {
  
  //jacobian of local parameters with respect to initial local parameters and material parameter xi
  //n.b this is the jacobian in LOCAL parameters (so E multiplies to the left of H s.t the total projection is E*Hprop*F)
  
  const double m2 = pow(updator.mass(), 2);  // use mass hypothesis from constructor
  constexpr double emass = 0.511e-3;
  constexpr double poti = 16.e-9 * 10.75;                 // = 16 eV * Z**0.9, for Si Z=14
  const double eplasma = 28.816e-9 * sqrt(2.33 * 0.498);  // 28.816 eV * sqrt(rho*(Z/A)) for Si
  const double qop = tsos.localParameters().qbp();
  const double dxdz = tsos.localParameters().dxdz();
  const double dydz = tsos.localParameters().dydz();
  const double xi = tsos.surface().mediumProperties().xi();

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
  const double x17 = 2*std::log(eplasma/poti) - std::log(x15*x16) - 1;
  const double x18 = x17*x4 + 2;
  const double x19 = std::pow(x5, -1.0/2.0);
  const double x20 = std::sqrt(std::pow(dxdz, 2) + std::pow(dydz, 2) + 1);
  const double x21 = x19*x20*xi;
  const double x22 = x18*x21;
  const double x23 = m2*qop;
  const double x24 = x0/std::pow(x22 + x0/qop, 2);
  const double x25 = x18*x19*x24;
  const double x26 = x25*xi/x20;
  const double res_0 = x24*(-x21*(-1.0/4.0*x1*x13*x4*x6*(x10*x16*x7*(-m2*x0 - x0*x9 - x12*x23/x11)/std::pow(x13, 2) + 4*x14*x3*x7*x9 - 8*x15/std::pow(qop, 3))/(x8*x9) + 2*x17*x23) - x22*x23*x5 + x3);
  const double res_1 = -dxdz*x26;
  const double res_2 = -dydz*x26;
  const double res_3 = -x20*x25;
  
  Matrix<double, 5, 6> EdE = Matrix<double, 5, 6>::Zero();
  //jacobian of q/p wrt local state parameters
  EdE(0,0) = res_0;
  EdE(0,1) = res_1;
  EdE(0,2) = res_2;
  EdE(1,1) = 1.;
  EdE(2,2) = 1.;
  EdE(3,3) = 1.;
  EdE(4,4) = 1.;
  //derivative of q/p wrt xi
  EdE(0,5) = res_3;
  
  return EdE;
}

std::array<Matrix<double, 5, 5>, 5> ResidualGlobalCorrectionMaker::processNoiseJacobians(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const {
  
  //this is the variation of the process noise matrix with respect to the relevant local parameters (qop, dxdz, dydz) and the material parameters xi and radlen
  
  const double m2 = pow(updator.mass(), 2);  // use mass hypothesis from constructor
  constexpr double emass = 0.511e-3;
  constexpr double logfact = 0.038;
  constexpr double amscon = 1.8496e-4;  // (13.6MeV)**2
  const double qop = tsos.localParameters().qbp();
  const double dxdz = tsos.localParameters().dxdz();
  const double dydz = tsos.localParameters().dydz();
  const double signpz = tsos.localParameters().pzSign();
  const double xi = tsos.surface().mediumProperties().xi();
  const double radLen = tsos.surface().mediumProperties().radLen();
    
  const double x0 = std::pow(qop, 5);
  const double x1 = std::pow(qop, 2);
  const double x2 = 1.0/x1;
  const double x3 = m2 + x2;
  const double x4 = 1.0/x3;
  const double x5 = x2*x4;
  const double x6 = 1 - 1.0/2.0*x5;
  const double x7 = std::pow(dxdz, 2);
  const double x8 = std::pow(dydz, 2);
  const double x9 = x7 + x8;
  const double x10 = x9 + 1;
  const double x11 = std::sqrt(x10);
  const double x12 = x11*xi;
  const double x13 = x12*x6;
  const double x14 = std::pow(x3, 2);
  const double x15 = std::pow(emass, 2);
  const double x16 = 1.0/m2;
  const double x17 = std::sqrt(x3);
  const double x18 = emass*x16;
  const double x19 = 2*x18;
  const double x20 = x15*x16 + x17*x19 + 1;
  const double x21 = 1.0/x20;
  const double x22 = x19*x21;
  const double x23 = x14*x22;
  const double x24 = x18*x21;
  const double x25 = 8*x24;
  const double x26 = std::fabs(qop);
  const double x27 = (((qop) > 0) - ((qop) < 0));
  const double x28 = 1.0/x27;
  const double x29 = x28/x26;
  const double x30 = x13*x29;
  const double x31 = std::pow(qop, 4);
  const double x32 = x3*x31;
  const double x33 = 14*x24;
  const double x34 = std::pow(x3, 3.0/2.0);
  const double x35 = std::pow(qop, 3);
  const double x36 = 1.0/x14;
  const double x37 = std::pow(qop, 7)*x23*x29;
  const double x38 = x37*x6;
  const double x39 = x38*xi/x11;
  const double x40 = std::pow(signpz, 2);
  const double x41 = x40/x10;
  const double x42 = x41*x7;
  const double x43 = x41*x8;
  const double x44 = x42 + x43;
  const double x45 = 1.0/x44;
  const double x46 = 1.0*x45;
  const double x47 = std::pow(x10, 2);
  const double x48 = 1.0/x47;
  const double x49 = std::pow(signpz, 4);
  const double x50 = x48*x49;
  const double x51 = x46*x50;
  const double x52 = x42*x46 + x51*x8;
  const double x53 = amscon*radLen;
  const double x54 = 1.0/x49;
  const double x55 = std::pow(x10, 5.0/2.0)*x54;
  const double x56 = x53*x55;
  const double x57 = x52*x56;
  const double x58 = radLen*x11;
  const double x59 = x29*x58;
  const double x60 = logfact*std::log(qop*x59) + 1;
  const double x61 = std::pow(x60, 2);
  const double x62 = x3*x35;
  const double x63 = 1.0*x62;
  const double x64 = x61*x63;
  const double x65 = x29*x61;
  const double x66 = x57*x65;
  const double x67 = 2.0*x1;
  const double x68 = amscon*x52;
  const double x69 = logfact*x60;
  const double x70 = x54*x69;
  const double x71 = 1.0/qop;
  const double x72 = 2.0*x3;
  const double x73 = x31*x72*(-x58*x71 + x59);
  const double x74 = x47*x70*x73;
  const double x75 = std::pow(x10, 3.0/2.0);
  const double x76 = x53*x75;
  const double x77 = dxdz*x76;
  const double x78 = x52*x77;
  const double x79 = x0*x29*x72;
  const double x80 = x70*x79;
  const double x81 = x3*x65;
  const double x82 = x0*x81;
  const double x83 = 5.0*x54*x82;
  const double x84 = 2.0*x45;
  const double x85 = dxdz*x84;
  const double x86 = std::pow(dxdz, 3);
  const double x87 = x40*x48;
  const double x88 = x84*x87;
  const double x89 = dxdz*x8;
  const double x90 = 4.0*x45*x49/std::pow(x10, 3);
  const double x91 = 2*x41;
  const double x92 = 2*x87;
  const double x93 = 1.0/std::pow(x44, 2);
  const double x94 = x93*(-dxdz*x91 + x86*x92 + x89*x92);
  const double x95 = x50*x8;
  const double x96 = 1.0*x82;
  const double x97 = x56*x96;
  const double x98 = dydz*x76;
  const double x99 = x80*x98;
  const double x100 = x83*x98;
  const double x101 = dydz*x84;
  const double x102 = std::pow(dydz, 3);
  const double x103 = dydz*x7;
  const double x104 = x93*(-dydz*x91 + x102*x92 + x103*x92);
  const double x105 = x55*x68;
  const double x106 = x69*x79;
  const double x107 = x26*x27;
  const double x108 = x107*x61;
  const double x109 = 1.0/x40;
  const double x110 = dxdz*dydz*x109;
  const double x111 = x110*x76;
  const double x112 = x1*x3;
  const double x113 = x112*x61;
  const double x114 = amscon*x110;
  const double x115 = x108*x63;
  const double x116 = x109*x115;
  const double x117 = amscon*x109*x58;
  const double x118 = x103*x117;
  const double x119 = x107*x60;
  const double x120 = 2.0*x62;
  const double x121 = logfact*x119*x120;
  const double x122 = 3.0*x108*x62;
  const double x123 = x117*x89;
  const double x124 = x114*x75;
  const double x125 = x43*x46 + x51*x7;
  const double x126 = x125*x56;
  const double x127 = amscon*x125;
  const double x128 = x125*x77;
  const double x129 = x50*x7;
  const double x130 = x127*x55;
  const double x131 = -x5;
  const double x132 = x131 + 1;
  const double x133 = qop*x3;
  const double x134 = x131 + 2;
  const double x135 = x132*x29;
  const double x136 = x134*x14;
  const double x137 = x112*x29;
  const double x138 = x136*x29;
  const double x139 = x134*x29;
  const double x140 = x1*x139;
  const double x141 = qop*x22;
  const double x142 = 4*x24;
  const double x143 = 4.0*x61;
  const double x144 = logfact*(x29 - x71);
  const double x145 = x144*x60;
  const double x146 = -8.0*qop*x145;
  const double x147 = 20.0*x62;
  const double x148 = 4.0*x112;
  const double x149 = x144*(-x119*x2 + x144 + x60*x71);
  const double x150 = x56*(-14.0*qop*x65 - 8.0*x113 + x143 + x145*x147 - x145*x148*x26*x28 + x146 + x147*x65 + x149*x79)/x9;
  const double x151 = 6.0*x108;
  const double delosdqop = std::pow(qop, 6)*x14*x30*x33 - x0*x13*x23 + x12*x37*(x4/x35 - x36/x0) - x25*x30*x32 + 4*x15*x30*x31*x34/(std::pow(m2, 2)*std::pow(x20, 2));
  const double delosddxdz = dxdz*x39;
  const double delosddydz = dydz*x39;
  const double delosdxi = x11*x38;
  const double delosdradLen = 0;
  const double dmsxxdqop = 5.0*x32*x66 - x57*x64 - x66*x67 + x68*x74;
  const double dmsxxddxdz = x78*x80 + x78*x83 + x97*(x41*x85 + x42*x94 - x86*x88 - x89*x90 + x94*x95);
  const double dmsxxddydz = x100*x52 + x52*x99 + x97*(x101*x50 - x102*x90 - x103*x88 + x104*x42 + x104*x95);
  const double dmsxxdxi = 0;
  const double dmsxxdradLen = x105*x106 + x105*x96;
  const double dmsxydqop = x10*x114*x69*x73 + 3.0*x107*x111*x113 - 2.0*x108*x111 + x111*x64;
  const double dmsxyddxdz = x116*x98 + x118*x121 + x118*x122;
  const double dmsxyddydz = x116*x77 + x121*x123 + x122*x123;
  const double dmsxydxi = 0;
  const double dmsxydradLen = x115*x124 + x121*x124;
  const double dmsyydqop = 5.0*x126*x31*x81 - x126*x64 - x126*x65*x67 + x127*x74;
  const double dmsyyddxdz = x128*x80 + x128*x83 + x97*(x129*x94 + x43*x94 + x50*x85 - x8*x85*x87 - x86*x90);
  const double dmsyyddydz = x100*x125 + x125*x99 + x97*(x101*x41 - x102*x88 - x103*x90 + x104*x129 + x104*x43);
  const double dmsyydxi = 0;
  const double dmsyydradLen = x106*x130 + x130*x96;
  const double d2elosdqop2 = x12*x141*(x1*x138*x24*(x142*x5 + x2/x34 - 3/x17) + 14*x112*x135 - 2*x132*x133 + 4*x133*x134 - 28*x134*x137 - x134*x141*x34 + x135*x142*x17 - 8*x135 - 6*x136*x35 - x137*(-7*x5 + 3 + 4*x36/x31) + 21*x138*x31 - x139*x17*x25 + x140*x33*x34 + 2*x140*(3*m2 + 5*x2));
  const double d2msxxdqop2 = x150*(x43 + x7);
  const double d2msxydqop2 = x111*(x107*x120*x149 + 6.0*x113 + x119*x144*x148 + x133*x151 - x143 + 12.0*x145*x62 + x146 - x151*x71);
  const double d2msyydqop2 = x150*(x42 + x8);
  
  std::array<Matrix<double, 5, 5>, 5> res;
  
  Matrix<double, 5, 5> &dQdqop = res[0];
  dQdqop = Matrix<double, 5, 5>::Zero();
  dQdqop(0,0) = delosdqop;
  dQdqop(1,1) = dmsxxdqop;
  dQdqop(1,2) = dmsxydqop;
  dQdqop(2,1) = dmsxydqop;
  dQdqop(2,2) = dmsyydqop;
  
//   std::cout << "dQdqop" << std::endl;
//   std::cout << dQdqop << std::endl;
  
//   Matrix<double, 5, 5> &d2Qdqop2 = res[1];
//   d2Qdqop2 = Matrix<double, 5, 5>::Zero();
//   d2Qdqop2(0,0) = d2elosdqop2;
//   d2Qdqop2(1,1) = d2msxxdqop2;
//   d2Qdqop2(1,2) = d2msxydqop2;
//   d2Qdqop2(2,1) = d2msxydqop2;
//   d2Qdqop2(2,2) = d2msyydqop2;
  
  Matrix<double, 5, 5> &dQddxdz = res[1];
  dQddxdz = Matrix<double, 5, 5>::Zero();
  dQddxdz(0,0) = delosddxdz;
  dQddxdz(1,1) = dmsxxddxdz;
  dQddxdz(1,2) = dmsxyddxdz;
  dQddxdz(2,1) = dmsxyddxdz;
  dQddxdz(2,2) = dmsyyddxdz;
  
  Matrix<double, 5, 5> &dQddydz = res[2];
  dQddydz = Matrix<double, 5, 5>::Zero();
  dQddydz(0,0) = delosddydz;
  dQddydz(1,1) = dmsxxddydz;
  dQddydz(1,2) = dmsxyddydz;
  dQddydz(2,1) = dmsxyddydz;
  dQddydz(2,2) = dmsyyddydz;
  
  Matrix<double, 5, 5> &dQdxi = res[3];
  dQdxi = Matrix<double, 5, 5>::Zero();
  dQdxi(0,0) = delosdxi;
  dQdxi(1,1) = dmsxxdxi;
  dQdxi(1,2) = dmsxydxi;
  dQdxi(2,1) = dmsxydxi;
  dQdxi(2,2) = dmsyydxi;
  
  Matrix<double, 5, 5> &dQdradLen = res[4];
  dQdradLen = Matrix<double, 5, 5>::Zero();
  dQdradLen(0,0) = delosdradLen;
  dQdradLen(1,1) = dmsxxdradLen;
  dQdradLen(1,2) = dmsxydradLen;
  dQdradLen(2,1) = dmsxydradLen;
  dQdradLen(2,2) = dmsyydradLen;
//   
  return res;
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

template <typename T>
void ResidualGlobalCorrectionMaker::init_twice_active_var(T &ad, const unsigned int d_num, const unsigned int idx) const {
  // initialize derivative direction in value field of outer active variable
  ad.value().derivatives() = T::DerType::Scalar::DerType::Unit(d_num,idx);
  // initialize derivatives direction of the variable
  ad.derivatives() = T::DerType::Unit(d_num,idx);
  // initialize Hessian matrix of variable to zero
  for(unsigned int idx=0;idx<d_num;idx++){
    ad.derivatives()(idx).derivatives()  = T::DerType::Scalar::DerType::Zero(d_num);
  }
}

template <typename T>
void ResidualGlobalCorrectionMaker::init_twice_active_null(T &ad, const unsigned int d_num) const {
  // initialize derivative direction in value field of outer active variable
  ad.value().derivatives() = T::DerType::Scalar::DerType::Zero(d_num);
  // initialize derivatives direction of the variable
  ad.derivatives() = T::DerType::Zero(d_num);
  // initialize Hessian matrix of variable to zero
  for(unsigned int idx=0;idx<d_num;idx++){
    ad.derivatives()(idx).derivatives()  = T::DerType::Scalar::DerType::Zero(d_num);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ResidualGlobalCorrectionMaker);
