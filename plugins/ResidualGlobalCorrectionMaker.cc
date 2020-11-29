// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

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

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"


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

class ResidualGlobalCorrectionMaker : public edm::stream::EDAnalyzer<>
{
public:
  explicit ResidualGlobalCorrectionMaker(const edm::ParameterSet &);
  ~ResidualGlobalCorrectionMaker();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  
  virtual void beginStream(edm::StreamID) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endStream() override;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  
  Matrix<double, 5, 6> localTransportJacobian(const TrajectoryStateOnSurface& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult,
                                              bool doReverse = false) const;
                                              
  Matrix<double, 5, 6> curv2localTransportJacobian(const FreeTrajectoryState& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult,
                                              bool doReverse = false) const;

  Matrix<double, 5, 6> curv2curvTransportJacobian(const FreeTrajectoryState& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult,
                                              bool doReverse = false) const;
                                              
  Matrix<double, 5, 6> curvtransportJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s,
                                                             const GlobalVector& bfield) const;
                                              
  Matrix<double, 5, 1> bfieldJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s,
                                                             const GlobalVector& bfield) const;
                                                             
  AlgebraicVector5 localMSConvolution(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const;
                                                             
  Matrix<double, 5, 6> materialEffectsJacobian(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator);
  
  std::array<Matrix<double, 5, 5>, 5> processNoiseJacobians(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const;
  
  Matrix<double, 2, 1> localPositionConvolution(const TrajectoryStateOnSurface& tsos) const;
  
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
//   edm::EDGetTokenT<std::vector<PSimHit>> inputSimHits_;
  std::vector<edm::EDGetTokenT<std::vector<PSimHit>>> inputSimHits_;

  TFile *fout;
  TTree *tree;
//   TTree *runtree;
//   TTree *gradtree;
//   TTree *hesstree;

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
  
  std::vector<unsigned int> hitidxv;
  std::vector<float> dxrecgen;
  std::vector<float> dyrecgen;
  std::vector<float> dxsimgen;
  std::vector<float> dysimgen;
  std::vector<float> dxrecsim;
  std::vector<float> dyrecsim;
  
  std::map<std::pair<int, DetId>, unsigned int> detidparms;
  
  unsigned int run;
  unsigned int lumi;
  unsigned long long event;
  
  std::vector<double> gradagg;
  
  std::unordered_map<std::pair<unsigned int, unsigned int>, double> hessaggsparse;
  
  bool fitFromGenParms_;
  bool fillTrackTree_;
  bool fillGrads_;
  
  bool debugprintout_;
  
  bool doSim_;
  
  float dxpxb1;
  float dypxb1;
  
  float dxttec9rphisimgen;
  float dyttec9rphisimgen;
  float dxttec9rphi;
  float dxttec9stereo;

  float dxttec4rphisimgen;
  float dyttec4rphisimgen;
  float dxttec4rphirecsim;
  
  float dxttec4rphi;
  float dxttec4stereo;
  
  float simlocalxref;
  float simlocalyref;
  
  float simtestz;
  float simtestzlocalref;
  float simtestdx;
  float simtestdxrec;
  
//   bool filledRunTree_;
  
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
  inputBs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  
  
  inputTrackOrig_ = consumes<reco::TrackCollection>(edm::InputTag(iConfig.getParameter<edm::InputTag>("src")));

  
  fitFromGenParms_ = iConfig.getParameter<bool>("fitFromGenParms");
  fillTrackTree_ = iConfig.getParameter<bool>("fillTrackTree");
  fillGrads_ = iConfig.getParameter<bool>("fillGrads");
  doSim_ = iConfig.getParameter<bool>("doSim");
  
  if (doSim_) {
//     inputSimHits_ = consumes<std::vector<PSimHit>>(edm::InputTag("g4SimHits","TrackerHitsTECLowTof"));
    std::vector<std::string> labels;
    labels.push_back("TrackerHitsPixelBarrelLowTof");
    labels.push_back("TrackerHitsPixelEndcapLowTof");
    labels.push_back("TrackerHitsTECLowTof");
    labels.push_back("TrackerHitsTIBLowTof");
    labels.push_back("TrackerHitsTIDLowTof");
    labels.push_back("TrackerHitsTOBLowTof");
    
    for (const std::string& label : labels) {
      inputSimHits_.push_back(consumes<std::vector<PSimHit>>(edm::InputTag("g4SimHits", label)));
    }
  }
  
  debugprintout_ = false;


//   fout = new TFile("trackTreeGrads.root", "RECREATE");
//   fout = new TFile("trackTreeGradsdebug.root", "RECREATE");
//   fout = new TFile("trackTreeGrads.root", "RECREATE");
  //TODO this needs a newer root version
//   fout->SetCompressionAlgorithm(ROOT::kLZ4);
//   fout->SetCompressionLevel(3);
  
//   edm::Service<TgFileService> fs;
  
//   tree = new TTree("tree", "tree");
  


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
  
  edm::ESHandle<TrackerTopology> trackerTopology;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopology);
  
//   ESHandle<MagneticField> magfield;
//   iSetup.get<IdealMagneticFieldRecord>().get(magfield);
//   auto field = magfield.product();
  
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

//   Handle<std::vector<PSimHit>> tecSimHits;
  std::vector<Handle<std::vector<PSimHit>>> simHits(inputSimHits_.size());
  if (doSim_) {
    for (unsigned int isimhit = 0; isimhit<inputSimHits_.size(); ++isimhit) {
      iEvent.getByToken(inputSimHits_[isimhit], simHits[isimhit]);
    }
  }
//   if (doSim_) {
//     iEvent.getByToken(inputSimHits_, tecSimHits);
//   }
  
//   const float mass = 0.105;
//   const float maxDPhi = 1.6;
//   PropagatorWithMaterial rPropagator(oppositeToMomentum, mass, field, maxDPhi, true, -1., false);
//   PropagatorWithMaterial fPropagator(alongMomentum, mass, field, maxDPhi, true, -1., false);
  
  std::unique_ptr<PropagatorWithMaterial> fPropagator(static_cast<PropagatorWithMaterial*>(thePropagator->clone()));
  fPropagator->setPropagationDirection(alongMomentum);
  
  std::unique_ptr<PropagatorWithMaterial> fAnalyticPropagator(static_cast<PropagatorWithMaterial*>(theAnalyticPropagator->clone()));
  fAnalyticPropagator->setPropagationDirection(alongMomentum);
  
  KFUpdator updator;
  TkClonerImpl const& cloner = static_cast<TkTransientTrackingRecHitBuilder const *>(ttrh.product())->cloner();

  // set up cylindrical surface for beam pipe
//   const double ABe = 9.0121831;
//   const double ZBe = 4.;
//   const double K =  0.307075*1e-3;
//   const double dr = 0.08;
// //   const double xibeampipe = 0.5*K*dr*ZBe/ABe;
//   const double xibeampipe = 0.*0.5*K*dr*ZBe/ABe;
  
  
  
  
//   auto beampipe = Cylinder::build(Surface::PositionType(0.,0.,0.), Surface::RotationType(), 2.94);
//   beampipe->setMediumProperties(MediumProperties(0., xibeampipe));
  
//   std::cout << "xi beampipe: " << xibeampipe << std::endl;
  
//   const GeomDet *testdet = nullptr;
//   //debugging
//   for (const GeomDet* det : globalGeometry->detUnits()) {
//     if (!det) {
//       continue;
//     }
//     
//     if (det->subDetector() == GeomDetEnumerators::TEC) {
//       const DetId& detid = det->geographicalId();
// //       TECDetId detid(det->geographicalId());
// //       layer = -1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel();
//       unsigned int side = trackerTopology->tecSide(detid);
//       unsigned int wheel = trackerTopology->tecWheel(detid);
//       int layer = -1 * (side == 1) * wheel + (side == 2) * wheel;
//       bool stereo = trackerTopology->isStereo(det->geographicalId());
//       
//       if (layer == -9) {
//         testdet = det;
//         break;
//         
//       }
//     }
//     
//     
//     
//   }
//   
//   if (testdet) {
//     const GlobalPoint center = testdet->surface().toGlobal(LocalPoint(1.,0.));
//     
//     const GlobalVector centerv(center.x(), center.y(), center.z());
//     const GlobalVector dir = centerv/centerv.mag();
//     const double sintheta = dir.perp();
//     const GlobalVector mom = (100000./sintheta)*dir;
//     const GlobalPoint pos(0.,0.,0.);
//     
//     FreeTrajectoryState ftsplus(pos, mom, 1., field);
//     FreeTrajectoryState ftsminus(pos, mom, -1., field);
//     
//     const TrajectoryStateOnSurface tsosplus = fPropagator->propagate(ftsplus, testdet->surface());
//     const TrajectoryStateOnSurface tsosminus = fPropagator->propagate(ftsminus, testdet->surface());
//     
//     std::cout << "global target" << std::endl;
//     std::cout << center << std::endl;
//     
//     std::cout << "momentum" << std::endl;
//     std::cout << mom << std::endl;
//     
//     std::cout << "tsosplus local:" << std::endl;
//     std::cout << tsosplus.localPosition() << std::endl;
//     std::cout << "tsosminus local:" << std::endl;
//     std::cout << tsosminus.localPosition() << std::endl;
//     
//     std::cout << "delta local" << std::endl;
//     std::cout << tsosplus.localPosition() - tsosminus.localPosition() << std::endl;
//     
//   }
  
  
  simtestzlocalref = -99.;
  simtestdx = -99.;
  simtestdxrec = -99.;
  
  if (false) {
    
    //sim hit debugging
    const reco::GenParticle* genmuon = nullptr;
    for (const reco::GenParticle& genPart : *genPartCollection) {
      if (genPart.status()==1 && std::abs(genPart.pdgId()) == 13) {
        genmuon = &genPart;
        break;
      }
    }
    
    if (genmuon) {
      genPt = genmuon->pt();
      genCharge = genmuon->charge();
      
      auto const& refpoint = genmuon->vertex();
      auto const& trackmom = genmuon->momentum();
      const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
      const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z()); 
  //     const GlobalTrajectoryParameters refglobal(refpos, refmom, genmuon->charge(), field);
      
  //       std::cout << "gen ref state" << std::endl;
  //       std::cout << refpos << std::endl;
  //       std::cout << refmom << std::endl;
  //       std::cout << genpart->charge() << std::endl;
      
      //zero uncertainty on generated parameters
  //       AlgebraicSymMatrix55 nullerr;
  //       const CurvilinearTrajectoryError referr(nullerr);
      
      const FreeTrajectoryState fts = FreeTrajectoryState(refpos, refmom, genmuon->charge(), field);
      
      std::cout << "gen muon charge: " << genmuon->charge() << std::endl;
      
      TrajectoryStateOnSurface tsos;
      
      unsigned int ihit = 0;
      for (auto const& simhith : simHits) {
        for (const PSimHit& simHit : *simhith) {
          
          const GeomDet *detectorG = globalGeometry->idToDet(simHit.detUnitId());
          
          bool isbarrel = detectorG->subDetector() == GeomDetEnumerators::PixelBarrel || detectorG->subDetector() == GeomDetEnumerators::TIB || detectorG->subDetector() == GeomDetEnumerators::TOB;
          
          float absz = std::abs(detectorG->surface().toGlobal(LocalVector(0.,0.,1.)).z());
          
          bool idealdisk = absz == 1.;
          bool idealbarrel = absz<1e-9;
          
  //         idealdisk = false;
  //         idealbarrel = false;
          
          bool isstereo = trackerTopology->isStereo(simHit.detUnitId());

          std::cout << "isbarrel: " << isbarrel << " idealbarrel: " << idealbarrel << " idealdisk: " << idealdisk << "stereo: " << isstereo << " globalpos: " << detectorG->surface().position() << std::endl;
          
          LocalPoint proplocal(0.,0.,0.);
          
//           auto const propresult = fPropagator->geometricalPropagator().propagateWithPath(fts, detectorG->surface());
//           if (propresult.first.isValid()) {
//             proplocal = propresult.first.localPosition();
//           }
          
          if (!tsos.isValid()) {
            tsos = fPropagator->geometricalPropagator().propagate(fts, detectorG->surface());
          }
          else {
            tsos = fPropagator->geometricalPropagator().propagate(tsos, detectorG->surface());
          }
          
          if (tsos.isValid()) {
            proplocal = tsos.localPosition();
          }
          
          
          
          
  //         Vector3d refprop;
          
  //         LocalTrajectoryParameters
          Point3DBase<double, LocalTag> reflocal(0, 0., 0.);

          simtestz = detectorG->surface().position().z();
          

          auto const simhitglobal = detectorG->surface().toGlobal(Point3DBase<double, LocalTag>(simHit.localPosition().x(),
                                                                                                simHit.localPosition().y(),
                                                                                                simHit.localPosition().z()));
          
          const Vector3d Msim(simhitglobal.x(), simhitglobal.y(), simhitglobal.z());
          
          auto const propglobal = detectorG->surface().toGlobal(Point3DBase<double, LocalTag>(proplocal.x(),
                                                                                                proplocal.y(),
                                                                                                proplocal.z()));
          
          
          const Vector3d Mprop(propglobal.x(), propglobal.y(), propglobal.z());
          
          Vector3d M(genmuon->vertex().x(),
                                  genmuon->vertex().y(),
                                  genmuon->vertex().z());
          
          Vector3d P(genmuon->momentum().x(),
                                  genmuon->momentum().y(),
                                  genmuon->momentum().z());
          
          
          
          
  //         if (true) {
          for (unsigned int iref=0; iref<1; ++iref) {
            const double zs = detectorG->surface().position().z();
            
            const Vector3d T0 = P.normalized();
            
  //           const Vector3d T0 = P.normalized();
            
            const Vector3d H(0.,0.,1.);
            
            const double rho = fts.transverseCurvature();
            
            double s;
            
            if (idealdisk) {
              s = (zs - M[2])/T0[2];
            }
            else if (false) {
              HelixBarrelPlaneCrossingByCircle crossing(GlobalPoint(M[0],M[1],M[2]), GlobalVector(P[0],P[1],P[2]), rho);
              s = crossing.pathLength(detectorG->surface()).second;
            }
            else {
              HelixArbitraryPlaneCrossing crossing(Basic3DVector<float>(M[0],M[1],M[2]), Basic3DVector<float>(P[0],P[1],P[2]), rho);
              s = crossing.pathLength(detectorG->surface()).second;
  //             s = propresult.second;
            }
            
            const Vector3d HcrossT = H.cross(T0);
            const double alpha = HcrossT.norm();
            const Vector3d N0 = HcrossT.normalized();
            
            const double gamma = T0[2];
            const double q = genmuon->charge();
            const double Q = -3.8*2.99792458e-3*q/P.norm();
            const double theta = Q*s;
            
            const Vector3d dM = gamma*(theta-std::sin(theta))/Q*H + std::sin(theta)/Q*T0 + alpha*(1.-std::cos(theta))/Q*N0;
            M = M + dM;
            const Vector3d dT = gamma*(1.-std::cos(theta))*H + std::cos(theta)*T0 + alpha*std::sin(theta)*N0;
            const Vector3d T = T0 + dT;
            const double pmag = P.norm();
            P = pmag*T;
            
            reflocal = detectorG->surface().toLocal(Point3DBase<double, GlobalTag>(M[0], M[1], M[2]));
            simtestzlocalref = reflocal.z();
            
            const Vector3d xhat = Vector3d(0.,0.,1.).cross(M).normalized();
            
            const double dx = xhat.dot(Msim-M);
            const double dxrec = xhat.dot(Mprop - Msim);
            
            simtestdx = dx;
  //           simtestdxrec = dxrec;
            simtestdxrec = proplocal.x() - simHit.localPosition().x();
            
            
            if (idealdisk) {
              break;
            }
            
  //           refprop = M;
            
  //           const Vector3d Mprop(updtsosnomat.globalPosition().x(),
  //                                 updtsosnomat.globalPosition().y(),
  //                                 updtsosnomat.globalPosition().z()); 
          }
          
          tree->Fill();
  //         else {
  //           const TrajectoryStateOnSurface propresult = fAnalyticPropagator->geometricalPropagator().propagate(fts, detectorG->surface());
  //           if (propresult.isValid()) {
  //             reflocal = propresult.localPosition();
  // //             refprop << propresult.globalPosition().x(), propresult.globalPosition().y(), propresult.globalPosition().z();
  //           }
  // //           else {
  // //             refprop << 0., 0., 0.;
  // //           }
  //         }
          
          
  //         const LocalPoint reflocal = detectorG->surface().toLocal(GlobalPoint(refprop[0], refprop[1], refprop[2]));
          

          
          const LocalPoint simlocal = simHit.localPosition();
          
//           std::cout << "isbarrel: " << isbarrel << " idealbarrel: " << idealbarrel << " idealdisk: " << idealdisk << "stereo: " << isstereo << " globalpos: " << detectorG->surface().position() << std::endl;
          std::cout << "detid: " << simHit.detUnitId() << std::endl;
          std::cout << "local z to global: " << detectorG->surface().toGlobal(LocalVector(0.,0.,1.)) << std::endl;
          std::cout << "ref      : " << reflocal << std::endl;
          std::cout << "proplocal: " << proplocal << std::endl;
          std::cout << "sim-ref : " << simlocal - reflocal << std::endl;
          std::cout << "prop-ref: " << proplocal - reflocal << std::endl;
          std::cout << "sim-prop: " << simlocal - proplocal << std::endl;
          
          
          ++ihit;
          
  //         if (simHit.detUnitId() == preciseHit->geographicalId()) {                      
  //           dxsimgen.push_back(simHit.localPosition().x() - updtsos.localPosition().x());
  //           dysimgen.push_back(simHit.localPosition().y() - updtsos.localPosition().y());
  //           
  //           dxrecsim.push_back(preciseHit->localPosition().x() - simHit.localPosition().x());
  //           dyrecsim.push_back(-99.);
  //           
  //           simvalid = true;
  //           break;
  //         }
        }
      }
      
      
    }
    return;
    
  }
  
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
    
    //prepare hits
    TransientTrackingRecHit::RecHitContainer hits;
    hits.reserve(track.recHitsSize());
//     hits.reserve(track.recHitsSize()+1);
//     hits.push_back(RecHitPointer(InvalidTrackingRecHit());
    
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
        hits.push_back((*it)->cloneForFit(*detectorG));
      }
    }
    
//     const unsigned int nhits = track.recHitsSize();
    const unsigned int nhits = hits.size();
    nHits = nhits;
//     unsigned int npixhits = 0;

    unsigned int nvalid = 0;
    unsigned int nvalidpixel = 0;
    unsigned int nvalidalign2d = 0;
    
    // count valid hits since this is needed to size the arrays
    for (auto const& hit : hits) {
      assert(hit->dimension()<=2);
      if (hit->isValid()) {
        nvalid += 1;
        
//         const uint32_t gluedid = trackerTopology->glued(hit->geographicalId());
//         const bool isglued = gluedid != 0;
//         const DetId parmdetid = isglued ? DetId(gluedid) : hit->geographicalId();
//         const bool align2d = detidparms.count(std::make_pair(1, parmdetid));
        const bool align2d = detidparms.count(std::make_pair(2, hit->geographicalId()));
        
        if (align2d) {
          nvalidalign2d += 1;
        }
        if (GeomDetEnumerators::isTrackerPixel(hit->det()->subDetector())) {
          nvalidpixel += 1;
        }
      }
    }
    
//     //count valid hits since this is needed to size the arrays
//     auto const& hitsbegin = track.recHitsBegin();
//     for (unsigned int ihit = 0; ihit < track.recHitsSize(); ++ihit) {
//       auto const& hit = *(hitsbegin + ihit);
//       if (hit->isValid() && hit->dimension()<=2) {
//         nvalid += 1;
//         
//         const GeomDet *detectorG = globalGeometry->idToDet(hit->geographicalId());
//         if (hit->dimension()==2 && GeomDetEnumerators::isTrackerPixel(detectorG->subDetector())) {
// //         if (hit->dimension()==2) {
//           nvalidpixel += 1;
//         }
//       }
//     }
    
    nValidHits = nvalid;
    nValidPixelHits = nvalidpixel;
    
//     const unsigned int nstriphits = nhits-npixhits;
//     const unsigned int nparsAlignment = nstriphits + 2*npixhits;
//     const unsigned int nvalidstrip = nvalid - nvalidpixel;
//     const unsigned int nparsAlignment = nvalidstrip + 2*nvalidpixel;
    const unsigned int nparsAlignment = 2*nvalid + nvalidalign2d;
    const unsigned int nparsBfield = nhits;
    const unsigned int nparsEloss = nhits - 1;
    const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
    
//     const unsigned int nstateparms = 5*(nhits+1);
    const unsigned int nstateparms = 3*(nhits+1) - 1;
//     const unsigned int nstateparms = 3*nhits - 1;
    const unsigned int nparmsfull = nstateparms + npars;
    
    
    const unsigned int nstateparmspost = 5*nhits;
    
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
//     using MSScalar = AANT<double, 11>;;
    using MSScalar = AANT<double, 13>;
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
    
    MatrixXd statejac;
    VectorXd dxstate;
    
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
    
    if (debugprintout_) {
      std::cout << "initial reference point parameters:" << std::endl;
      std::cout << track.parameters() << std::endl;
    }

//     //prepare hits
//     TransientTrackingRecHit::RecHitContainer hits;
//     hits.reserve(track.recHitsSize());
//     for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
//       const GeomDet *detectorG = globalGeometry->idToDet((*it)->geographicalId());
//       hits.push_back((*it)->cloneForFit(*detectorG));
//     }
    
//     // fix mixed up clusters?
//     for (unsigned int ihit=0; ihit<(hits.size()-1); ++ihit) {
//       TrackerSingleRecHit* hit = const_cast<TrackerSingleRecHit*>(dynamic_cast<const TrackerSingleRecHit*>(hits[ihit].get()));
//       TrackerSingleRecHit* nexthit = const_cast<TrackerSingleRecHit*>(dynamic_cast<const TrackerSingleRecHit*>(hits[ihit+1].get()));
//       
// //      const TrackingRecHitSingle* nexthit = hits[ihit+1];
//       
//       if (!hit || !nexthit) {
//         continue;
//       }
//       
//       const DetId partnerid = trackerTopology->partnerDetId(hit->geographicalId());
// //       
//       if (partnerid == nexthit->geographicalId()) {
// //         std::cout << "swapping clusters" << std::endl;
//         const OmniClusterRef::ClusterStripRef cluster = hit->cluster_strip();
//         const OmniClusterRef::ClusterStripRef nextcluster = nexthit->cluster_strip();
//         
//         hit->setClusterStripRef(nextcluster);
//         nexthit->setClusterStripRef(cluster);
//       }
// 
//     }
    
    
    FreeTrajectoryState refFts;
    
    if (dogen) {
      //init from gen state
      auto const& refpoint = genpart->vertex();
      auto const& trackmom = genpart->momentum();
      const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
      const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z()); 
      const GlobalTrajectoryParameters refglobal(refpos, refmom, genpart->charge(), field);
      
//       std::cout << "gen ref state" << std::endl;
//       std::cout << refpos << std::endl;
//       std::cout << refmom << std::endl;
//       std::cout << genpart->charge() << std::endl;
      
      //zero uncertainty on generated parameters
//       AlgebraicSymMatrix55 nullerr;
//       const CurvilinearTrajectoryError referr(nullerr);
      
      refFts = FreeTrajectoryState(refpos, refmom, genpart->charge(), field);
//       refFts = FreeTrajectoryState(refglobal, referr);
    }
    else {
      //init from track state
      auto const& refpoint = track.referencePoint();
      auto const& trackmom = track.momentum();
      const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
      const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z()); 
      const GlobalTrajectoryParameters refglobal(refpos, refmom, track.charge(), field);
//       const CurvilinearTrajectoryError referr(track.covariance());
      
      //null uncertainty (tracking process noise sum only)
//       AlgebraicSymMatrix55 nullerr;
//       const CurvilinearTrajectoryError referr(nullerr);
      
      refFts = FreeTrajectoryState(refpos, refmom, track.charge(), field);
//       refFts = FreeTrajectoryState(refglobal, referr);
    }

//     std::vector<std::pair<TrajectoryStateOnSurface, double>> layerStates;
    std::vector<TrajectoryStateOnSurface> layerStates;
    layerStates.reserve(nhits);
    
    bool valid = true;

    
//     //do propagation and prepare states
//     auto propresult = fPropagator->propagateWithPath(refFts, *hits.front()->surface());
//     if (!propresult.first.isValid()) {
//       std::cout << "Abort: Propagation from reference point failed" << std::endl;
//       continue;
//     }
//     layerStates.push_back(propresult);
//     
//     for (auto const& hit : hits) {
//       propresult = fPropagator->propagateWithPath(layerStates.back().first, *hit->surface());
//       if (!propresult.first.isValid()) {
//         std::cout << "Abort: Propagation failed" << std::endl;
//         valid = false;
//         break;
//       }
//       layerStates.push_back(propresult);
//     }
//     
//     if (!valid) {
//       continue;
//     }
    

    
    //inflate errors
//     refFts.rescaleError(100.);
    
    
//     unsigned int ntotalhitdim = 0;
//     unsigned int alignmentidx = 0;
//     unsigned int bfieldidx = 0;
//     unsigned int elossidx = 0;
    
//     constexpr unsigned int niters = 2;
    constexpr unsigned int niters = 1;
    
    for (unsigned int iiter=0; iiter<niters; ++iiter) {
      if (debugprintout_) {
        std::cout<< "iter " << iiter << std::endl;
      }
      
      hitidxv.clear();
      hitidxv.reserve(nvalid);
      
      dxrecgen.clear();
      dxrecgen.reserve(nvalid);
      
      dyrecgen.clear();
      dyrecgen.reserve(nvalid);
      
      dxsimgen.clear();
      dxsimgen.reserve(nvalid);
      
      dysimgen.clear();
      dysimgen.reserve(nvalid);
      
      dxrecsim.clear();
      dxrecsim.reserve(nvalid);
      
      dyrecsim.clear();
      dyrecsim.reserve(nvalid);
      
        
      const bool islikelihood = iiter > 0;
//       const bool islikelihood = true;
      
      gradfull = VectorXd::Zero(nparmsfull);
      hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);
      statejac = MatrixXd::Zero(nstateparmspost, nstateparms);
      
      evector<std::array<Matrix<double, 8, 8>, 11> > dhessv;
      if (islikelihood) {
        dhessv.resize(nhits-1);
      }
      
      dxpxb1 = -99.;
      dypxb1 = -99.;
      dxttec9rphi = -99.;
      dxttec9stereo = -99.;
      dxttec4rphi = -99.;
      dxttec4stereo = -99.;
      
      dxttec4rphisimgen = -99.;
      dyttec4rphisimgen = -99.;
      dxttec4rphirecsim = -99.;
      
      dxttec9rphisimgen = -99.;
      dyttec9rphisimgen = -99.;
      
      simlocalxref = -99.;
      simlocalyref = -99.;
      
      
//       dhessv.reserve(nhits-1);
      
      unsigned int parmidx = 0;
      unsigned int alignmentparmidx = 0;

      if (iiter > 0) {
        //update current state from reference point state (errors not needed beyond first iteration)
        JacobianCurvilinearToCartesian curv2cart(refFts.parameters());
        const AlgebraicMatrix65& jac = curv2cart.jacobian();
        const AlgebraicVector6 glob = refFts.parameters().vector();
        
        auto const& dxlocal = dxstate.head<5>();
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
//         currentFts = refFts;
      }
      
//       Matrix5d Hlm = Matrix5d::Identity();
//       currentFts = refFts;
//       TrajectoryStateOnSurface currentTsos;

//       ;
      
      auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, *hits[0]->surface());
//       auto propresult = fPropagator->geometricalPropagator().propagateWithPath(refFts, *beampipe);
      if (!propresult.first.isValid()) {
        std::cout << "Abort: Propagation of reference state Failed!" << std::endl;
        valid = false;
        break;
      }
      
//       std::cout << "position on beampipe " << propresult.first.globalParameters().position() << std::endl;
      
      const Matrix<double, 5, 6> FdFp = curv2curvTransportJacobian(refFts, propresult, false);

      Matrix<double, 2, 2> J = FdFp.block<2, 2>(3, 3);
      // (du/dalphap)^-1
      Matrix<double, 2, 2> Sinv = FdFp.block<2, 2>(3, 1).inverse();
      // du/dqopp
      Matrix<double, 2, 1> D = FdFp.block<2, 1>(3, 0);
      // du/dBp
      Matrix<double, 2, 1> Bpref = FdFp.block<2, 1>(3, 5);

      constexpr unsigned int jacstateidxout = 0;
      constexpr unsigned int jacstateidxin = 0;
      
      // qop_i
      statejac(jacstateidxout, jacstateidxin + 2) = 1.;
      // d(lambda, phi)_i/dqop_i
      statejac.block<2, 1>(jacstateidxout + 1, jacstateidxin + 2) = -Sinv*D;
      // d(lambda, phi)_i/(dxy, dsz)
      statejac.block<2, 2>(jacstateidxout + 1, jacstateidxin) = -Sinv*J;
      // d(lambda, phi)_i/du_(i+1)
      statejac.block<2, 2>(jacstateidxout + 1, jacstateidxin + 3) = Sinv;
      // dxy
      statejac(jacstateidxout + 3, jacstateidxin) = 1.;
      // dsz
      statejac(jacstateidxout + 4, jacstateidxin + 1) = 1.;
      
      Matrix<double, 5, 6> FdFm = curv2curvTransportJacobian(refFts, propresult, true);
      
      for (unsigned int ihit = 0; ihit < hits.size(); ++ihit) {
//         std::cout << "ihit " << ihit << std::endl;
        auto const& hit = hits[ihit];
                
        TrajectoryStateOnSurface updtsos = propresult.first;
        
        //apply measurement update if applicable
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
        
        const bool align2d = detidparms.count(std::make_pair(2, preciseHit->geographicalId()));

        
        // compute convolution correction in local coordinates (BEFORE material effects are applied)
//         const Matrix<double, 2, 1> dxlocalconv = localPositionConvolution(updtsos);
         
        // curvilinear to local jacobian
        JacobianCurvilinearToLocal curv2localm(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
        const AlgebraicMatrix55& curv2localjacm = curv2localm.jacobian();
        const Matrix<double, 5, 5> Hm = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjacm.Array()); 
        
        //energy loss jacobian
        const Matrix<double, 5, 6> EdE = materialEffectsJacobian(updtsos, fPropagator->materialEffectsUpdator());
       
        //process noise jacobians
        const std::array<Matrix<double, 5, 5>, 5> dQs = processNoiseJacobians(updtsos, fPropagator->materialEffectsUpdator());
        
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
        

        //FIXME take care of this elsewhere for the moment
        const bool genconstraint = dogen && ihit==0;
//         const bool genconstraint = false;
        
        if (ihit < (nhits-1)) {

          //momentum kink residual
          AlgebraicVector5 idx0(0., 0., 0., 0., 0.);
          if (iiter==0) {
            layerStates.push_back(updtsos);
          }
          else {
            //FIXME this is not valid for the updated parameterization
            
            //current state from previous state on this layer
            //save current parameters          
            TrajectoryStateOnSurface& oldtsos = layerStates[ihit];
            
            const AlgebraicVector5 local = oldtsos.localParameters().vector();
            auto const& dxlocal = dxstate.segment<5>(5*(ihit+1));
            const Matrix<double, 5, 1> localupd = Map<const Matrix<double, 5, 1>>(local.Array()) + dxlocal;
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
          
          const Vector5d dx0 = Map<const Vector5d>(idx0.Array());
          

//           if (ihit==0) {
//             FreeTrajectoryState tmpfts(updtsospos, updtsos.globalParameters().momentum(), updtsos.charge(), field);
//             propresult = fPropagator->geometricalPropagator().propagateWithPath(tmpfts, *hits[ihit+1]->surface());
//           }
//           else {
//             propresult = fPropagator->geometricalPropagator().propagateWithPath(updtsos, *hits[ihit+1]->surface());
//           }
          propresult = fPropagator->geometricalPropagator().propagateWithPath(updtsos, *hits[ihit+1]->surface());
          if (!propresult.first.isValid()) {
            std::cout << "Abort: Propagation Failed!" << std::endl;
            valid = false;
            break;
          }
          
          if (true) {
//           if (false) {
            //forward propagation jacobian (local to local)
            const Matrix<double, 5, 6> FdFp = curv2curvTransportJacobian(*updtsos.freeState(), propresult, false);

            Matrix<double, 2, 2> J = FdFp.block<2, 2>(3, 3);
            // (du/dalphap)^-1
            Matrix<double, 2, 2> Sinv = FdFp.block<2, 2>(3, 1).inverse();
            // du/dqopp
            Matrix<double, 2, 1> D = FdFp.block<2, 1>(3, 0);
            
            const unsigned int jacstateidxout = 5*(ihit+1);
            const unsigned int jacstateidxin = 3*(ihit+1);
            
            // qop_i
            statejac(jacstateidxout, jacstateidxin + 2) = 1.;
            // dalpha_i/dqop_i
            statejac.block<2, 1>(jacstateidxout + 1, jacstateidxin + 2) = -Sinv*D;
            // dalpha_i/du_i
            statejac.block<2, 2>(jacstateidxout + 1, jacstateidxin) = -Sinv*J;
            // dalpha_i/du_(i+1)
            statejac.block<2, 2>(jacstateidxout + 1, jacstateidxin + 3) = Sinv;
            // xlocal_i
            statejac(jacstateidxout + 3, jacstateidxin) = 1.;
            // ylocal_i
            statejac(jacstateidxout + 4, jacstateidxin + 1) = 1.;
            
            
            
//             std::cout << "FdFm" << std::endl;
//             std::cout << FdFm << std::endl;
//             std::cout << "FdFp" << std::endl;
//             std::cout << FdFp << std::endl;
            
            constexpr unsigned int nlocalstate = 8;
            constexpr unsigned int nlocalbfield = 3;
            constexpr unsigned int nlocaleloss = 2;
            constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
            
            constexpr unsigned int nlocal = nlocalstate + nlocalbfield + nlocaleloss;
            
            constexpr unsigned int localstateidx = 0;
  //           constexpr unsigned int localbfieldidx = localstateidx + nlocalstate;
  //           constexpr unsigned int localelossidx = localbfieldidx + nlocalbfield;
            constexpr unsigned int localparmidx = localstateidx + nlocalstate;
            
            const unsigned int fullstateidx = 3*ihit;
//             const unsigned int fullstateidx = 3*(ihit-1);
            const unsigned int fullparmidx = (nstateparms + parmidx) - 2;
            
//             std::cout << "ihit = " << ihit << " nstateparms = " << nstateparms << " parmidx = " << parmidx << " fullparmidx = " << fullparmidx << std::endl;
             
            // individual pieces, now starting to cast to active scalars for autograd,
            // as in eq (3) of https://doi.org/10.1016/j.cpc.2011.03.017
            // du/dum
            Matrix<MSScalar, 2, 2> Jm = FdFm.block<2, 2>(3, 3).cast<MSScalar>();
            // (du/dalpham)^-1
            Matrix<MSScalar, 2, 2> Sinvm = FdFm.block<2, 2>(3, 1).inverse().cast<MSScalar>();
            // du/dqopm
            Matrix<MSScalar, 2, 1> Dm = FdFm.block<2, 1>(3, 0).cast<MSScalar>();
            // du/dBm
            Matrix<MSScalar, 2, 1> Bm = FdFm.block<2, 1>(3, 5).cast<MSScalar>();

            // du/dup
            Matrix<MSScalar, 2, 2> Jp = FdFp.block<2, 2>(3, 3).cast<MSScalar>();
            // (du/dalphap)^-1
            Matrix<MSScalar, 2, 2> Sinvp = FdFp.block<2, 2>(3, 1).inverse().cast<MSScalar>();
            // du/dqopp
            Matrix<MSScalar, 2, 1> Dp = FdFp.block<2, 1>(3, 0).cast<MSScalar>();
            // du/dBp
            Matrix<MSScalar, 2, 1> Bp = FdFp.block<2, 1>(3, 5).cast<MSScalar>();
            
//             std::cout << "Jm" << std::endl;
//             std::cout << Jm << std::endl;
//             std::cout << "Sinvm" << std::endl;
//             std::cout << Sinvm << std::endl;
//             std::cout << "Dm" << std::endl;
//             std::cout << Dm << std::endl;
//             std::cout << "Bm" << std::endl;
//             std::cout << Bm << std::endl;
//             
//             std::cout << "Jp" << std::endl;
//             std::cout << Jp << std::endl;
//             std::cout << "Sinvp" << std::endl;
//             std::cout << Sinvp << std::endl;
//             std::cout << "Dp" << std::endl;
//             std::cout << Dp << std::endl;
//             std::cout << "Bp" << std::endl;
//             std::cout << Bp << std::endl;
            
            // energy loss jacobians
  //           const MSJacobian E = EdE.leftCols<5>().cast<MSScalar>();
  //           const MSVector dE = EdE.rightCols<1>().cast<MSScalar>();
            
            // fraction of material on this layer compared to glued layer if relevant
//             double xifraction = isglued ? preciseHit->det()->surface().mediumProperties().xi()/parmDet->surface().mediumProperties().xi() : 1.;
            
//             std::cout << "xifraction: " << xifraction << std::endl;
            
            const MSScalar Eqop(EdE(0,0));
            const Matrix<MSScalar, 1, 2> Ealpha = EdE.block<1, 2>(0, 1).cast<MSScalar>();
            const MSScalar dE(EdE(0,5));
//             const MSScalar dE(xifraction*EdE(0,5));
//             (void)EdE;
            
            const MSScalar muE(dxeloss[0]);
            
//             std::cout<<"EdE" << std::endl;
//             std::cout << EdE << std::endl;
            
            //energy loss inverse variance
            MSScalar invSigmaE(1./Q(0,0));
            
            // multiple scattering inverse covariance
            Matrix<MSScalar, 2, 2> Qinvms = Q.block<2,2>(1,1).inverse().cast<MSScalar>();
                        
            // initialize active scalars for state parameters
            Matrix<MSScalar, 2, 1> dum = Matrix<MSScalar, 2, 1>::Zero();
            //suppress gradients of reference point parameters when fitting with gen constraint
            for (unsigned int j=0; j<dum.size(); ++j) {
              init_twice_active_var(dum[j], nlocal, localstateidx + j);
              //FIXME this would be the correct condition if we were using it
//               if (dogen && ihit < 2) {
//               if (genconstraint) {
//                 init_twice_active_null(dum[j], nlocal);
//               }
//               else {
//                 init_twice_active_var(dum[j], nlocal, localstateidx + j);
//               }
            }

            
            MSScalar dqopm(0.);
            init_twice_active_var(dqopm, nlocal, localstateidx + 2);
            
//             //suppress gradients of reference point parameters when fitting with gen constraint
//             if (genconstraint) {
//               init_twice_active_null(dqopm, nlocal);
//             }
//             else {
//               init_twice_active_var(dqopm, nlocal, localstateidx + 2);
//             }

            Matrix<MSScalar, 2, 1> du = Matrix<MSScalar, 2, 1>::Zero();
            for (unsigned int j=0; j<du.size(); ++j) {
              init_twice_active_var(du[j], nlocal, localstateidx + 3 + j);
//               if (genconstraint) {
//                 init_twice_active_null(du[j], nlocal);
//               }
//               else {
//                 init_twice_active_var(du[j], nlocal, localstateidx + 3 + j);
//               }
            }
            
            MSScalar dqop(0.);
            init_twice_active_var(dqop, nlocal, localstateidx + 5);

            Matrix<MSScalar, 2, 1> dup = Matrix<MSScalar, 2, 1>::Zero();
            for (unsigned int j=0; j<dup.size(); ++j) {
              init_twice_active_var(dup[j], nlocal, localstateidx + 6 + j);
            }
  
            // initialize active scalars for correction parameters
            
            // only used for gen constraint
            MSScalar dbetam(0.);
            
            MSScalar dbeta(0.);
            init_twice_active_var(dbeta, nlocal, localparmidx + 2);
            
            MSScalar dxi(0.);
            init_twice_active_var(dxi, nlocal, localparmidx + 3);
            
            MSScalar dbetap(0.);
            init_twice_active_var(dbetap, nlocal, localparmidx + 4);
            
            if (dogen && ihit==0) {
              du = Bpref.cast<MSScalar>()*dbeta;
            }
            else if (dogen && ihit==1) {
              init_twice_active_var(dbetam, nlocal, localparmidx);
              dum = Bpref.cast<MSScalar>()*dbetam;
            }
            
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
            
            
//             const Matrix<MSScalar, 2, 1> dalpham = Sinvm*(dum - Jm*du - Dm*dqopm - Bm*dbeta);
//             const Matrix<MSScalar, 2, 1> dalphap = Sinvp*(dup - Jp*du - Dp*dqop - Bp*dbetap);
//             const Matrix<MSScalar, 2, 1> dalpham = Sinvm*(dum - Jm*du - Dm*dqopm);
//             const Matrix<MSScalar, 2, 1> dalphap = Sinvp*(dup - Jp*du - Dp*dqop);
            
            
            const MSScalar deloss0(dx0[0]);

            
            
            
//             const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
//             const MSScalar chisqms = dms.transpose()*Qinvms*dms;
//             //energy loss term
//             
//             
//             const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
//             const MSScalar chisqeloss = deloss*deloss*invSigmaE;
//             
//             const MSScalar chisq = chisqms + chisqeloss;
            
            

            
//             const bool dolikelihood = false;
//           
            MSScalar chisq;
            
            if (!islikelihood) {
              //standard chisquared contribution
              
              const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
              const MSScalar chisqms = dms.transpose()*Qinvms*dms;
              //energy loss term
              
              
              const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
              const MSScalar chisqeloss = deloss*invSigmaE*deloss;
              
              chisq = chisqms + chisqeloss;
            }
            else {
//               islikelihood = true;
              //maximum likelihood contribution 
              const MSCovariance dQdqop = dQs[0].cast<MSScalar>();
//               const MSCovariance dQddxdz = dQs[1].cast<MSScalar>();
//               const MSCovariance dQddydz = dQs[2].cast<MSScalar>();
//               const MSCovariance dQdxi = dQs[3].cast<MSScalar>();
              
//               const MSCovariance dQ = dqopm*dQdqop + dalpham[0]*dQddxdz + dalpham[1]*dQddydz + dxi*dQdxi;
//               const MSCovariance dQ = 0.5*(dqop+dqopm)*dQdqop;
              const MSCovariance dQ = dqopm*dQdqop;
//               const MSCovariance dQ = 0.5*(dqop+dqopm)*dQdqop + 0.5*(dalpham[0] + dalphap[0])*dQddxdz + 0.5*(dalpham[1]+dalphap[1])*dQddydz + dxi*dQdxi;
              
              const Matrix<MSScalar, 2, 2> Qmsnom = Q.block<2,2>(1,1).cast<MSScalar>();
              const Matrix<MSScalar, 2, 2> Qmsnominv = Qmsnom.inverse();
              const Matrix<MSScalar, 2, 2> Qmsinv = Qmsnominv - Qmsnominv*dQ.block<2,2>(1,1)*Qmsnominv;
              
              
//               const Matrix<MSScalar, 2, 2> Qms = Q.block<2,2>(1,1).cast<MSScalar>() + dQ.block<2,2>(1,1);
//               const Matrix<MSScalar, 2, 2> Qmsinv = Qms.inverse();
//               const MSScalar logdetQms = Eigen::log(Qms.determinant());
              
              const Matrix<MSScalar, 2, 1> dms = dalpha0 + dalphap - dalpham;
              MSScalar chisqms = dms.transpose()*Qmsinv*dms;
//               chisqms = chisqms + logdetQms;
              
              //energy loss term
//               const MSScalar sigmaE = MSScalar(Q(0,0)) + dQ(0,0);
//               const MSScalar sigmaEinv = 1./sigmaE;
              
              const MSScalar sigmaEnom = MSScalar(Q(0,0));
              const MSScalar sigmaEnominv = 1./sigmaEnom;
              
              const MSScalar sigmaEinv = sigmaEnominv - sigmaEnominv*dQ(0,0)*sigmaEnominv;
              
//               const MSScalar logsigmaE = Eigen::log(sigmaE);
              
              const MSScalar deloss = deloss0 + dqop - Eqop*dqopm - (Ealpha*dalpham)[0] - dE*dxi;
              MSScalar chisqeloss = deloss*sigmaEinv*deloss;
//               chisqeloss = chisqeloss + logsigmaE;
              
              chisq = chisqms + chisqeloss;
              
              //compute contributions to hessian matrix-vector derivative
              for (unsigned int i=0; i<nlocal; ++i) {
                MSScalar x(0.);
                init_twice_active_var(x, nlocal, i);
                
                Matrix<MSScalar, 2, 2> dQmsinv;
                for (unsigned int j=0; j<2; ++j) {
                  for (unsigned int k=0; k<2; ++k) {
                    dQmsinv(j,k) = MSScalar(Qmsinv(j,k).value().derivatives()[i]);
                  }
                }
                const MSScalar dSigmaEinv(sigmaEinv.value().derivatives()[i]);
                
                MSScalar dchisqms = dms.transpose()*dQmsinv*x*dms;
//                 dchisqms = 3.*dchisqms;
                MSScalar dchisqeloss = deloss*deloss*dSigmaEinv*x;
//                 dchisqeloss = 3.*dchisqeloss;
                const MSScalar dchisq = dchisqms + dchisqeloss;
                
                //TODO should this be 11x11 instead?
                //TODO check additional factor of 2
                for (unsigned int j=0; j<8; ++j) {
                  for (unsigned int k=0; k<8; ++k) {
                    dhessv[ihit][i](j,k) = dchisq.derivatives()[j].derivatives()[k];
                  }
                }
                
              }
              
            }
            
          
            
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
            
            const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(3,preciseHit->geographicalId()));
            globalidxv[parmidx] = bfieldglobalidx;
            parmidx++;
            
            const unsigned int elossglobalidx = detidparms.at(std::make_pair(4,preciseHit->geographicalId()));
            globalidxv[parmidx] = elossglobalidx;
            parmidx++;
          }
                    
          //backwards propagation jacobian (local to local) to be used at the next layer
          FdFm = curv2curvTransportJacobian(*updtsos.freeState(), propresult, true);
          
        }
        else {
          const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(3,preciseHit->geographicalId()));
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

            using AlignScalar = AANT<double, nlocal>;
            
            //FIXME dirty hack to abuse state idx for reference point magnetic field
            const unsigned int fullstateidx = genconstraint ? nstateparms : 3*(ihit+1);
  //           const unsigned int fullstateidx = 3*ihit;
            const unsigned int fullparmidx = nstateparms + nparsBfield + nparsEloss + alignmentparmidx;

            const bool ispixel = GeomDetEnumerators::isTrackerPixel(preciseHit->det()->subDetector());

            //TODO add hit validation stuff
            //TODO add simhit stuff

            Matrix<AlignScalar, 2, 2> Hu = Hp.bottomRightCorner<2,2>().cast<AlignScalar>();

            Matrix<AlignScalar, 2, 1> dy0;
            Matrix<AlignScalar, 2, 2> Vinv;
            // rotation from module to strip coordinates
            Matrix<AlignScalar, 2, 2> R;
            if (preciseHit->dimension() == 1) {
              dy0[0] = AlignScalar(preciseHit->localPosition().x() - updtsos.localPosition().x());
              dy0[1] = AlignScalar(0.);
              
              Vinv = Matrix<AlignScalar, 2, 2>::Zero();
              Vinv(0,0) = 1./preciseHit->localPositionError().xx();
              
              R = Matrix<AlignScalar, 2, 2>::Identity();
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
                
                R = Matrix<AlignScalar, 2, 2>::Identity();
              }
              else {
                // diagonalize and take only smallest eigenvalue for 2d hits in strip wedge modules,
                // since the constraint parallel to the strip is spurious
                SelfAdjointEigenSolver<Matrix2d> eigensolver(iV);
                const Matrix2d& v = eigensolver.eigenvectors();
                
                Matrix<double, 2, 1> dy0local;
                dy0local[0] = preciseHit->localPosition().x() - updtsos.localPosition().x();
                dy0local[1] = preciseHit->localPosition().y() - updtsos.localPosition().y();
                
                const Matrix<double, 2, 1> dy0eig = v.transpose()*dy0local;
                
                //TODO deal properly with rotations (rotate back to module local coords?)
                dy0[0] = AlignScalar(dy0eig[0]);
                dy0[1] = AlignScalar(0.);
                
                Vinv = Matrix<AlignScalar, 2, 2>::Zero();
                Vinv(0,0) = AlignScalar(1./eigensolver.eigenvalues()[0]);      
                
                R = v.transpose().cast<AlignScalar>();
              }
            }
            
            Matrix<AlignScalar, 2, 1> dx = Matrix<AlignScalar, 2, 1>::Zero();
            AlignScalar dbeta(0.);
            if (!genconstraint) {
              for (unsigned int j=0; j<dx.size(); ++j) {
                init_twice_active_var(dx[j], nlocal, localstateidx + j);
              }
            }
            else {
              init_twice_active_var(dbeta, nlocal, localstateidx);
              dx = Bpref.cast<AlignScalar>()*dbeta;
            }

            Matrix<AlignScalar, 3, 1> dalpha = Matrix<AlignScalar, 3, 1>::Zero();
            for (unsigned int idim=0; idim<nlocalalignment; ++idim) {
              init_twice_active_var(dalpha[idim], nlocal, localalignmentidx+idim);
            }
            
            // alignment jacobian
            Matrix<AlignScalar, 2, 3> A = Matrix<AlignScalar, 2, 3>::Zero();
            // dx/dtheta
            A(0,0) = -updtsos.localPosition().y();
            // dy/dtheta
            A(1,0) = updtsos.localPosition().x();
            // dx/dx
            A(0,1) = AlignScalar(1.);
            // dy/dy
            A(1,2) = AlignScalar(1.);
            

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

            Matrix<AlignScalar, 2, 1> dh = dy0 - R*Hu*dx - R*A*dalpha;
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
              const unsigned int xglobalidx = detidparms.at(std::make_pair(idim, preciseHit->geographicalId()));
              globalidxv[nparsBfield + nparsEloss + alignmentparmidx] = xglobalidx;
              alignmentparmidx++;
              if (idim==0) {
                hitidxv.push_back(xglobalidx);
              }
            }
            
            // fill hit validation information
            dxrecgen.push_back(dy0[0].value().value());
            dyrecgen.push_back(dy0[1].value().value());

            if (doSim_) {
              bool simvalid = false;
              for (auto const& simhith : simHits) {
                for (const PSimHit& simHit : *simhith) {
                  if (simHit.detUnitId() == preciseHit->geographicalId()) {                      
                    dxsimgen.push_back(simHit.localPosition().x() - updtsos.localPosition().x());
                    dysimgen.push_back(simHit.localPosition().y() - updtsos.localPosition().y());
                    
                    dxrecsim.push_back(preciseHit->localPosition().x() - simHit.localPosition().x());
                    dyrecsim.push_back(preciseHit->localPosition().y() - simHit.localPosition().y());
                    
                    simvalid = true;
                    break;
                  }
                }
                if (simvalid) {
                  break;
                }
              }
              if (!simvalid) {
                dxsimgen.push_back(-99.);
                dysimgen.push_back(-99.);
                dxrecsim.push_back(-99.);
                dyrecsim.push_back(-99.);
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
        
//         std::cout << "hit " << ihit << " isvalid " << preciseHit->isValid() << std::endl;
//         std::cout << "global position: " << updtsos.globalParameters().position() << std::endl;
        //hit information
        //FIXME consolidate this special cases into templated function(s)
//         if (preciseHit->isValid()) {
      }
            
      if (!valid) {
        break;
      }
      
      assert(parmidx == (nparsBfield + nparsEloss));
      assert(alignmentparmidx == nparsAlignment);
      
      //fake constraint on reference point parameters
      if (dogen) {
//       if (false) {
        for (unsigned int i=0; i<5; ++i) {
          gradfull[i] = 0.;
          hessfull.row(i) *= 0.;
          hessfull.col(i) *= 0.;
          hessfull(i,i) = 1e6;
        }
//         //b field from reference point not consistently used in this case
//         gradfull[nstateparms] = 0.;
//         hessfull.row(nstateparms) *= 0.;
//         hessfull.col(nstateparms) *= 0.;
      }
      
      //now do the expensive calculations and fill outputs
      auto const& dchisqdx = gradfull.head(nstateparms);
      auto const& dchisqdparms = gradfull.tail(npars);
      
      auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
      auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
      auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
      
//       std::cout << "dchisqdx" << std::endl;
//       std::cout << dchisqdx << std::endl;
//       std::cout << "d2chisqdx2 diagonal" << std::endl;
//       std::cout << d2chisqdx2.diagonal() << std::endl;
//       std::cout << "d2chisqdx2" << std::endl;
//       std::cout << d2chisqdx2 << std::endl;
//       
//       auto const& eigenvalues = d2chisqdx2.eigenvalues();
//       std::cout << "d2chisqdx2 eigenvalues" << std::endl;
//       std::cout << eigenvalues << std::endl;
      
//       auto const& Cinvd = d2chisqdx2.ldlt();
      Cinvd.compute(d2chisqdx2);
      
      
      if (islikelihood) {
        const MatrixXd Cfull = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms));
        
        // add ln det terms to gradient and hessian
  //       MatrixXd dhessfulli;
  //       MatrixXd dhessfullj;
  //       VectorXd dgradfull;
        // TODO should this cover correction parameter part of the matrix as well?
        for (unsigned int ihit=0; ihit<(nhits-1); ++ihit) {
          constexpr unsigned int localstateidx = 0;
          const unsigned int fullstateidx = 3*ihit;
          
          auto const& Cblock = Cfull.block<8,8>(fullstateidx, fullstateidx);
          
  //         dhessfulli = MatrixXd::Zero(nstateparms, nstateparms);
  //         dhessfullj = MatrixXd::Zero(nstateparms, nstateparms);
          
          //TODO fill correction parameter block as well
          for (unsigned int i=0; i<8; ++i) {
            gradfull[fullstateidx + i] += (Cblock*dhessv[ihit][i]).trace();
            for (unsigned int j=0; j<8; ++j) {
              hessfull(fullstateidx + i, fullstateidx + j) += (-Cblock*dhessv[ihit][j]*Cblock*dhessv[ihit][i]).trace();
            }
          }
          
        }
        
        Cinvd.compute(d2chisqdx2);
      
      }
      
      dxfull = -Cinvd.solve(dchisqdx);
      
      dxstate = statejac*dxfull;
      
//       const Vector5d dxRef = dx.head<5>();
// //       const Vector5d dxRef = -Cinvd.solve(dchisqdx).head<5>();
//       const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();
      
//       dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
//       
//       grad = dchisqdparms + dxdparms*dchisqdx;
//       hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
//       
//       std::cout << "dxfull" << std::endl;
//       std::cout << dxfull << std::endl;
//       std::cout << "errsq" << std::endl;
//       std::cout << Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).diagonal() << std::endl;
      
      const Vector5d dxRef = dxstate.head<5>();
      const Matrix5d Cinner = (statejac*Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))*statejac.transpose()).topLeftCorner<5,5>();
      
//       const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();

      if (debugprintout_) {
        std::cout<< "dxRef" << std::endl;
        std::cout<< dxRef << std::endl;
      }
      
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
      
//       std::cout << "refParms" << std::endl;
//       std::cout << Map<const Vector5f>(refParms.data()) << std::endl;
      
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
    
//     if (debugprintout_) {
//       std::cout << "dxrefdparms" << std::endl;
//       std::cout << dxdparms.leftCols<5>() << std::endl;
//     }
    
    grad = dchisqdparms + dxdparms*dchisqdx;
    //TODO check the simplification
//     hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
    hess = d2chisqdparms2 + dxdparms*d2chisqdxdparms;
    
//     const Vector5d dxRef = dxfull.head<5>();
//     const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();


    
    gradv.clear();
    jacrefv.clear();

    gradv.resize(npars,0.);
    jacrefv.resize(5*npars, 0.);
    
    nJacRef = 5*npars;
    if (fillTrackTree_ && fillGrads_) {
      tree->SetBranchAddress("gradv", gradv.data());
    }
    if (fillTrackTree_) {
      tree->SetBranchAddress("jacrefv", jacrefv.data());
    }
    
    //eigen representation of the underlying vector storage
    Map<VectorXf> gradout(gradv.data(), npars);
    Map<Matrix<float, 5, Dynamic, RowMajor> > jacrefout(jacrefv.data(), 5, npars);
    
//     jacrefout = dxdparms.leftCols<5>().transpose().cast<float>();    
    jacrefout = (dxdparms*statejac.transpose()).leftCols<5>().transpose().cast<float>();  
    
    gradout = grad.cast<float>();
    
    
    float refPt = dogen ? genpart->pt() : std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1]);

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
    
//     if (gradmax < 1e5 && refPt > 5.5) {
//       //fill aggregrate gradient and hessian
//       for (unsigned int i=0; i<npars; ++i) {
//         gradagg[globalidxv[i]] += grad[i];
//       }
//       
//       hessmax = 0.;
//       for (unsigned int i=0; i<npars; ++i) {
//         for (unsigned int j=i; j<npars; ++j) {
//           const unsigned int iidx = globalidxv[i];
//           const unsigned int jidx = globalidxv[j];
//           
//           const float absval = std::abs(hess(i,j));
//           if (absval>hessmax) {
//             hessmax = absval;
//           }
//           
//           const std::pair<unsigned int, unsigned int> key = std::make_pair(std::min(iidx,jidx), std::max(iidx,jidx));
//           
//           auto it = hessaggsparse.find(key);
//           if (it==hessaggsparse.end()) {
//             hessaggsparse[key] = hess(i,j);
//           }
//           else {
//             it->second += hess(i,j);
//           }
//         }
//       }
//     }
    
    
    if (debugprintout_) {
      const Matrix5d Cinner = (statejac*Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms))*statejac.transpose()).topLeftCorner<5,5>();
      std::cout << "hess debug" << std::endl;
      std::cout << "track parms" << std::endl;
      std::cout << tkparms << std::endl;
  //     std::cout << "dxRef" << std::endl;
  //     std::cout << dxRef << std::endl;
      std::cout << "original cov" << std::endl;
      std::cout << track.covariance() << std::endl;
      std::cout << "recomputed cov" << std::endl;
      std::cout << 2.*Cinner << std::endl;
    }

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
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ResidualGlobalCorrectionMaker::beginStream(edm::StreamID streamid)
{
  std::stringstream filenamestream;
  filenamestream << "globalcor_" << streamid.value() << ".root";
  fout = new TFile(filenamestream.str().c_str(), "RECREATE");
  
//   runtree = new TTree("runtree","");
//   gradtree = fs->make<TTree>("gradtree","");
//   hesstree = fs->make<TTree>("hesstree","");
  
  
  if (fillTrackTree_) {
    tree = new TTree("tree","");
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
    
    tree->Branch("globalidxv", globalidxv.data(), "globalidxv[nParms]/i", basketSize);
    tree->Branch("jacrefv",jacrefv.data(),"jacrefv[nJacRef]/F", basketSize);
    
    if (fillGrads_) {
      tree->Branch("gradv", gradv.data(), "gradv[nParms]/F", basketSize);
      tree->Branch("nSym", &nSym, basketSize);
      tree->Branch("hesspackedv", hesspackedv.data(), "hesspackedv[nSym]/F", basketSize);
    }
    
    tree->Branch("run", &run);
    tree->Branch("lumi", &lumi);
    tree->Branch("event", &event);
    
    tree->Branch("gradmax", &gradmax);
    tree->Branch("hessmax", &hessmax);
    
//     tree->Branch("dxpxb1", &dxpxb1);
//     tree->Branch("dypxb1", &dypxb1);
//     
//     tree->Branch("dxttec9rphi", &dxttec9rphi);
//     tree->Branch("dxttec9stereo", &dxttec9stereo);
//     
//     tree->Branch("dxttec4rphi", &dxttec4rphi);
//     tree->Branch("dxttec4stereo", &dxttec4stereo);
//     
//     tree->Branch("dxttec4rphisimgen", &dxttec4rphisimgen);
//     tree->Branch("dyttec4rphisimgen", &dyttec4rphisimgen);
//     tree->Branch("dxttec4rphirecsim", &dxttec4rphirecsim);
//     
//     tree->Branch("dxttec9rphisimgen", &dxttec9rphisimgen);
//     tree->Branch("dyttec9rphisimgen", &dyttec9rphisimgen);
//     
//     tree->Branch("simlocalxref", &simlocalxref);
//     tree->Branch("simlocalyref", &simlocalyref);
    
    tree->Branch("hitidxv", &hitidxv);
    tree->Branch("dxrecgen", &dxrecgen);
    tree->Branch("dyrecgen", &dyrecgen);
    tree->Branch("dxsimgen", &dxsimgen);
    tree->Branch("dysimgen", &dysimgen);
    tree->Branch("dxrecsim", &dxrecsim);
    tree->Branch("dyrecsim", &dyrecsim);
    
    tree->Branch("simtestz", &simtestz);
    tree->Branch("simtestzlocalref", &simtestzlocalref);
    tree->Branch("simtestdx", &simtestdx);
    tree->Branch("simtestdxrec", &simtestdxrec);
    
    
    nParms = 0.;
    nJacRef = 0.;
    
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void ResidualGlobalCorrectionMaker::endStream()
{
  fout->cd();
  
//   TTree *gradtree = new TTree("gradtree","");
//   unsigned int idx;
//   double gradval;
//   gradtree->Branch("idx",&idx);
//   gradtree->Branch("gradval",&gradval);
//   for (unsigned int i=0; i<gradagg.size(); ++i) {
//     idx = i;
//     gradval = gradagg[i];
//     gradtree->Fill();
//   }
//   
//   TTree *hesstree = new TTree("hesstree","");
//   unsigned int iidx;
//   unsigned int jidx;
//   double hessval;
//   hesstree->Branch("iidx",&iidx);
//   hesstree->Branch("jidx",&jidx);
//   hesstree->Branch("hessval",&hessval);
//   
//   for (auto const& item : hessaggsparse) {
//     iidx = item.first.first;
//     jidx = item.first.second;
//     hessval = item.second;
//     hesstree->Fill();
//   }
  
  fout->Write();
  fout->Close();
}

// ------------ method called when starting to processes a run  ------------

void 
ResidualGlobalCorrectionMaker::beginRun(edm::Run const& run, edm::EventSetup const& es)
{
  if (detidparms.size()>0) {
    return;
  }
  
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  es.get<GlobalTrackingGeometryRecord>().get(globalGeometry);
  
  edm::ESHandle<TrackerTopology> trackerTopology;
  es.get<TrackerTopologyRcd>().get(trackerTopology);
  
  detidparms.clear();
  
  std::set<std::pair<int, DetId> > parmset;
  
  for (const GeomDet* det : globalGeometry->detUnits()) {
    if (!det) {
      continue;
    }
    if (GeomDetEnumerators::isTracker(det->subDetector())) {
      
//       std::cout << "detid: " << det->geographicalId().rawId() << std::endl;
      
//       std::cout << "detid: " << det->geographicalId().rawId() << " subdet: " << det->subDetector() << " isStereo: " << trackerTopology->isStereo(det->geographicalId()) << " isRphi: " << trackerTopology->isRPhi(det->geographicalId()) << " glued: " << trackerTopology->glued(det->geographicalId()) << " stack: " << trackerTopology->stack(det->geographicalId()) << " upper: " << trackerTopology->upper(det->geographicalId()) << " lower: " << trackerTopology->lower(det->geographicalId()) << " partner: " << trackerTopology->partnerDetId(det->geographicalId()).rawId() <<" xi: " << det->surface().mediumProperties().xi() << std::endl;

      const bool ispixel = GeomDetEnumerators::isTrackerPixel(det->subDetector());
      const bool isendcap = GeomDetEnumerators::isEndcap(det->subDetector());

      
//       const uint32_t gluedid = trackerTopology->glued(det->geographicalId());
//       const bool isglued = gluedid != 0;
// //       const bool align2d = ispixel || isglued || isendcap;      
//       const DetId parmdetid = isglued ? DetId(gluedid) : det->geographicalId();
      
      const bool align2d = ispixel || isendcap;

      
      //always have parameters for local x alignment, bfield, and e-loss
      parmset.emplace(0, det->geographicalId());
      parmset.emplace(1, det->geographicalId());
      if (align2d) {
        //local y alignment parameters only for pixels and disks for now
        parmset.emplace(2, det->geographicalId());
      }
      parmset.emplace(3, det->geographicalId());
      parmset.emplace(4, det->geographicalId());
    }
  }
  
//   assert(0);
  
//   TFile *runfout = new TFile("trackTreeGradsParmInfo.root", "RECREATE");
  fout->cd();
  TTree *runtree = new TTree("runtree", "");
  
  unsigned int iidx;
  int parmtype;
  unsigned int rawdetid;
  int subdet;
  int layer;
  int stereo;
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
  runtree->Branch("stereo", &stereo);
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
    stereo = 0;
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
      stereo = trackerTopology->isStereo(det->geographicalId());
    }
    else if (det->subDetector() == GeomDetEnumerators::TOB)
    {
//       TOBDetId detid(det->geographicalId());
//       layer = detid.layer();
      layer = trackerTopology->tobLayer(det->geographicalId());
      stereo = trackerTopology->isStereo(det->geographicalId());
    }
    else if (det->subDetector() == GeomDetEnumerators::TID)
    {
      unsigned int side = trackerTopology->tidSide(detid);
      unsigned int wheel = trackerTopology->tidWheel(detid);
      layer = -1 * (side == 1) * wheel + (side == 2) * wheel;
      stereo = trackerTopology->isStereo(det->geographicalId());

    }
    else if (det->subDetector() == GeomDetEnumerators::TEC)
    {
//       TECDetId detid(det->geographicalId());
//       layer = -1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel();
      unsigned int side = trackerTopology->tecSide(detid);
      unsigned int wheel = trackerTopology->tecWheel(detid);
      layer = -1 * (side == 1) * wheel + (side == 2) * wheel;
      stereo = trackerTopology->isStereo(det->geographicalId());
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
                                                             const double& s,
                                                             const GlobalVector& bfield) const {
 
  //analytic jacobian wrt magnitude of magnetic field
  //TODO should we parameterize with respect to z-component instead?
  //extending derivation from CMS NOTE 2006/001
  const Vector3d b(bfield.x(),
                      bfield.y(),
                      bfield.z());
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
  auto const xf25 = std::pow(xf24, 2);
  auto const xf26 = xf0[1];
  auto const xf27 = xf7[1];
  auto const xf28 = xf27*xf4;
  auto const xf29 = xf10[1];
  auto const xf30 = xf13*xf29;
  auto const xf31 = xf28 - xf30;
  auto const xf32 = xf26*xf6 + xf31;
  auto const xf33 = std::pow(xf32, 2);
  auto const xf34 = xf17*xf25 + xf17*xf33;
  auto const xf35 = 1.0/xf34;
  auto const xf36 = xf17*xf35;
  auto const xf37 = 1.0/(std::pow(xf16, 2)*xf36 + 1);
  auto const xf38 = qop*s;
  auto const xf39 = xf12*xf8;
  auto const xf40 = xf12*xf38;
  auto const xf41 = gamma*xf1;
  auto const xf42 = xf38*xf4;
  auto const xf43 = alpha*xf11;
  auto const xf44 = 1.0/std::fabs(qop);
  auto const xf45 = std::pow(xf34, -1.0/2.0);
  auto const xf46 = xf44*xf45;
  auto const xf47 = xf19*xf40;
  auto const xf48 = alpha*xf21;
  auto const xf49 = xf42*xf48;
  auto const xf50 = gamma*xf40;
  auto const xf51 = xf18*xf50;
  auto const xf52 = (1.0/2.0)*xf17;
  auto const xf53 = xf24*xf52;
  auto const xf54 = xf27*xf40;
  auto const xf55 = alpha*xf29;
  auto const xf56 = xf42*xf55;
  auto const xf57 = xf26*xf50;
  auto const xf58 = xf32*xf52;
  auto const xf59 = xf16*xf44/std::pow(xf34, 3.0/2.0);
  auto const xf60 = 1.0/magb;
  auto const xf61 = s*xf4;
  auto const xf62 = xf60*xf61;
  auto const xf63 = 1.0/qop;
  auto const xf64 = xf63/std::pow(magb, 2);
  auto const xf65 = xf12*xf64;
  auto const xf66 = xf62 - xf65;
  auto const xf67 = -gamma*xf62 + gamma*xf65;
  auto const xf68 = -xf12*xf3 + xf5;
  auto const xf69 = xf64*xf68;
  auto const xf70 = -xf16*(xf1*xf67 + xf43*xf69 + xf66*xf8) - xf24*(xf18*xf67 + xf19*xf66 + xf48*xf69) - xf32*(xf26*xf67 + xf27*xf66 + xf55*xf69);
  auto const xf71 = xf12*xf2;
  auto const xf72 = xf2*xf4;
  auto const xf73 = xf19*xf71;
  auto const xf74 = xf48*xf72;
  auto const xf75 = gamma*xf71;
  auto const xf76 = xf18*xf75;
  auto const xf77 = xf27*xf71;
  auto const xf78 = xf55*xf72;
  auto const xf79 = xf26*xf75;
  auto const xf80 = xf37*(xf46*(-xf2*xf39 + xf41*xf71 - xf43*xf72) + xf59*(-xf53*(-2*xf73 - 2*xf74 + 2*xf76) - xf58*(-2*xf77 - 2*xf78 + 2*xf79)));
  auto const xf81 = xf24*xf36;
  auto const xf82 = xf32*xf36;
  auto const xf83 = xf81*(-xf77 - xf78 + xf79) - xf82*(-xf73 - xf74 + xf76);
  auto const xf84 = U;
  auto const xf85 = xf84[0];
  auto const xf86 = s*xf60;
  auto const xf87 = xf5*xf64;
  auto const xf88 = gamma*xf18;
  auto const xf89 = xf12 - xf3;
  auto const xf90 = xf64*xf89;
  auto const xf91 = xf60*xf63;
  auto const xf92 = xf91*(-xf38 + xf42);
  auto const xf93 = -xf19*xf65 + xf20*xf86 - xf22*xf86 + xf48*xf87 + xf88*xf90 - xf88*xf92;
  auto const xf94 = xf84[1];
  auto const xf95 = gamma*xf26;
  auto const xf96 = -xf27*xf65 + xf28*xf86 - xf30*xf86 + xf55*xf87 + xf90*xf95 - xf92*xf95;
  auto const xf97 = xf84[2];
  auto const xf98 = -xf14*xf86 + xf41*xf90 - xf41*xf92 + xf43*xf87 - xf65*xf8 + xf86*xf9;
  auto const xf99 = xf91*(-xf2 + xf72);
  auto const xf100 = xf23 - xf88*xf99;
  auto const xf101 = xf31 - xf95*xf99;
  auto const xf102 = xf15 - xf41*xf99;
  auto const xf103 = xf100*xf85 + xf101*xf94 + xf102*xf97;
  auto const xf104 = V;
  auto const xf105 = xf104[0];
  auto const xf106 = xf104[1];
  auto const xf107 = xf104[2];
  auto const xf108 = xf100*xf105 + xf101*xf106 + xf102*xf107;
  auto const xf109 = xf17*(((qop) > 0) - ((qop) < 0));
  auto const xf110 = magb*s;
  auto const xf111 = xf110*xf12;
  auto const xf112 = xf110*xf4;
  auto const xf113 = std::pow(qop, -3);
  auto const xf114 = xf111*xf19;
  auto const xf115 = xf112*xf48;
  auto const xf116 = xf111*xf88;
  auto const xf117 = xf111*xf27;
  auto const xf118 = xf112*xf55;
  auto const xf119 = xf111*xf95;
  auto const xf120 = xf61*xf63;
  auto const xf121 = xf17*xf60;
  auto const xf122 = xf12*xf121;
  auto const xf123 = xf120 - xf122;
  auto const xf124 = -gamma*xf120 + gamma*xf122;
  auto const xf125 = xf121*xf68;
  auto const xf126 = -xf16*(xf1*xf124 + xf123*xf8 + xf125*xf43) - xf24*(xf123*xf19 + xf124*xf18 + xf125*xf48) - xf32*(xf123*xf27 + xf124*xf26 + xf125*xf55);
  auto const xf127 = xf35*xf44;
  auto const xf128 = s*xf63;
  auto const xf129 = xf121*xf5;
  auto const xf130 = xf121*xf89;
  auto const xf131 = xf91*(-xf110 + xf112);
  auto const xf132 = -xf122*xf19 + xf128*xf20 - xf128*xf22 + xf129*xf48 + xf130*xf88 - xf131*xf88;
  auto const xf133 = -xf122*xf27 + xf128*xf28 - xf128*xf30 + xf129*xf55 + xf130*xf95 - xf131*xf95;
  auto const xf134 = -xf122*xf8 - xf128*xf14 + xf128*xf9 + xf129*xf43 + xf130*xf41 - xf131*xf41;
  auto const dlamdB = xf37*(xf46*(-xf38*xf39 + xf40*xf41 - xf42*xf43) + xf59*(-xf53*(-2*xf47 - 2*xf49 + 2*xf51) - xf58*(-2*xf54 - 2*xf56 + 2*xf57))) + xf70*xf80;
  auto const dphidB = xf70*xf83 + xf81*(-xf54 - xf56 + xf57) - xf82*(-xf47 - xf49 + xf51);
  auto const dxtdB = xf103*xf70 + xf85*xf93 + xf94*xf96 + xf97*xf98;
  auto const dytdB = xf105*xf93 + xf106*xf96 + xf107*xf98 + xf108*xf70;
  auto const dlamdqop = xf126*xf80 + xf37*(-xf109*xf16*xf45 + xf46*(-xf110*xf39 + xf111*xf41 - xf112*xf43) + xf59*(xf113*xf25 + xf113*xf33 - xf53*(-2*xf114 - 2*xf115 + 2*xf116) - xf58*(-2*xf117 - 2*xf118 + 2*xf119)));
  auto const dphidqop = xf126*xf83 + xf127*xf24*(-xf109*xf32 + xf44*(-xf117 - xf118 + xf119)) - xf127*xf32*(-xf109*xf24 + xf44*(-xf114 - xf115 + xf116));
  auto const dxtdqop = xf103*xf126 + xf132*xf85 + xf133*xf94 + xf134*xf97;
  auto const dytdqop = xf105*xf132 + xf106*xf133 + xf107*xf134 + xf108*xf126;


  Matrix<double, 5, 1> dF;
  dF[0] = 0.;
  dF[1] = dlamdB;
  dF[2] = dphidB;
  dF[3] = dxtdB;
  dF[4] = dytdB;
  
//   convert to tesla
  dF *= 2.99792458e-3;
  
  Matrix<double, 5, 1> Fqop;
  Fqop[0] = 1.;
  Fqop[1] = dlamdqop;
  Fqop[2] = dphidqop;
  Fqop[3] = dxtdqop;
  Fqop[4] = dytdqop;
// //   
//   std::cout << "Fqop from sympy:" << std::endl;
//   std::cout << Fqop << std::endl;
  
  return dF;
}

Matrix<double, 5, 6> ResidualGlobalCorrectionMaker::curvtransportJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s,
                                                             const GlobalVector& bfield) const {
   
  Matrix<double, 5, 6> res;

  const double qop0 = globalSource.signedInverseMomentum();
  const double limit = 5.;
  const double cutCriterion = std::abs(s * qop0);
  
//   std::cout << "cutCriterion " << cutCriterion << std::endl;
//   if (cutCriterion > limit) {
  if (true) {
    
    // large s limit, use CMSSW calculation which
    // explicitly uses final position and momentum from propagation
    
//     std::cout << "computing jacobian from cmssw" << std::endl;
    AnalyticalCurvilinearJacobian curv2curv;
    curv2curv.computeFullJacobian(globalSource, globalDest.position(), globalDest.momentum(), bfield, s);
    const AlgebraicMatrix55& curv2curvjac = curv2curv.jacobian();
    const Matrix<double, 5, 5> F = Map<const Matrix<double, 5, 5, RowMajor>>(curv2curvjac.Array());
    
    res.topLeftCorner<5,5>() = F;
    res(0,5) = 0.;
    res.block<4, 1>(1,5) = qop0/bfield.mag()*F.block<4, 1>(1, 0);
  }
  else {
    // small s limit, use sympy-based calculation which
    // uses final position and momentum implicitly from path length + transport equations
    
//     std::cout << "computing jacobian from sympy" << std::endl;
    
    const double M0x = globalSource.position().x();
    const double M0y = globalSource.position().y();
    const double M0z = globalSource.position().z();
    
    const double p0 = globalSource.momentum().mag();
    const GlobalVector W0 = globalSource.momentum()/p0;
    const double W0x = W0.x();
    const double W0y = W0.y();
    const double W0z = W0.z();
    
    const double B = bfield.mag();
    const GlobalVector H = bfield/B;
    const double hx = H.x();
    const double hy = H.y();
    const double hz = H.z();
    
      
    const double x0 = B*s;
    const double x1 = qop0*x0;
    const double x2 = std::cos(x1);
    const double x3 = std::pow(W0z, 2);
    const double x4 = std::pow(W0x, 2);
    const double x5 = std::pow(W0y, 2);
    const double x6 = x4 + x5;
    const double x7 = 1.0/x6;
    const double x8 = std::pow(x3*x7 + 1, -1.0/2.0);
    const double x9 = x2*x8;
    const double x10 = std::sqrt(x6);
    const double x11 = 1.0/x10;
    const double x12 = W0z*x11;
    const double x13 = x12*x9;
    const double x14 = W0y*hx;
    const double x15 = x11*x14;
    const double x16 = W0x*hy;
    const double x17 = x11*x16;
    const double x18 = x15 - x17;
    const double x19 = std::sin(x1);
    const double x20 = x19*x8;
    const double x21 = x18*x20;
    const double x22 = x2 - 1;
    const double x23 = hx*x8;
    const double x24 = W0x*x11;
    const double x25 = hy*x8;
    const double x26 = W0y*x11;
    const double x27 = hz*x8;
    const double x28 = x12*x27 + x23*x24 + x25*x26;
    const double x29 = x22*x28;
    const double x30 = -hz*x29 + x13 - x21;
    const double x31 = x26*x9;
    const double x32 = x24*x27;
    const double x33 = x12*x23 - x32;
    const double x34 = x19*x33;
    const double x35 = hy*x29;
    const double x36 = x31 + x34 - x35;
    const double x37 = x24*x9;
    const double x38 = x12*x25;
    const double x39 = x26*x27;
    const double x40 = x38 - x39;
    const double x41 = x19*x40;
    const double x42 = -hx*x29 + x37 - x41;
    const double x43 = std::pow(x36, 2) + std::pow(x42, 2);
    const double x44 = 1.0/x43;
    const double x45 = 1.0/(std::pow(x30, 2)*x44 + 1);
    const double x46 = x12*x20;
    const double x47 = -x15 + x17;
    const double x48 = x47*x9;
    const double x49 = hz*x28;
    const double x50 = x19*x49;
    const double x51 = std::pow(x43, -1.0/2.0);
    const double x52 = x20*x26;
    const double x53 = x0*x52;
    const double x54 = x0*x2;
    const double x55 = x33*x54;
    const double x56 = hy*x28;
    const double x57 = x0*x19;
    const double x58 = x56*x57;
    const double x59 = (1.0/2.0)*x36;
    const double x60 = x20*x24;
    const double x61 = x0*x60;
    const double x62 = -x38 + x39;
    const double x63 = x54*x62;
    const double x64 = hx*x28;
    const double x65 = x57*x64;
    const double x66 = (1.0/2.0)*x42;
    const double x67 = x30/std::pow(x43, 3.0/2.0);
    const double x68 = x22*x8;
    const double x69 = x18*x68;
    const double x70 = B*qop0;
    const double x71 = W0z*x24;
    const double x72 = W0z*x26;
    const double x73 = -M0x*x71 - M0y*x72 + M0z*(x11*x4 + x11*x5);
    const double x74 = x10*x73;
    const double x75 = x1 - x19;
    const double x76 = x46 + x49*x75 + x69 + x70*x74;
    const double x77 = 1.0/B;
    const double x78 = x77/std::pow(qop0, 2);
    const double x79 = x76*x78;
    const double x80 = x0 - x54;
    const double x81 = 1.0/qop0;
    const double x82 = x77*x81;
    const double x83 = x82*(B*x74 + x0*x13 - x0*x21 + x49*x80);
    const double x84 = -M0x*x26 + M0y*x24;
    const double x85 = W0z*x73;
    const double x86 = W0x*x84 - W0y*x85;
    const double x87 = B*x86;
    const double x88 = qop0*x87 + x10*(-x22*x33 + x52 + x56*x75);
    const double x89 = x11*x78;
    const double x90 = x88*x89;
    const double x91 = x10*(x0*x31 + x0*x34 + x56*x80) + x87;
    const double x92 = x11*x82;
    const double x93 = x91*x92;
    const double x94 = W0x*x85 + W0y*x84;
    const double x95 = B*x94;
    const double x96 = -qop0*x95 + x10*(x22*x40 + x60 + x64*x75);
    const double x97 = x89*x96;
    const double x98 = x10*(x0*x37 - x0*x41 + x64*x80) - x95;
    const double x99 = x92*x98;
    const double x100 = -x30*(-x79 + x83) - x36*(-x90 + x93) - x42*(-x97 + x99);
    const double x101 = x52*x70;
    const double x102 = x2*x70;
    const double x103 = x102*x33;
    const double x104 = x19*x70;
    const double x105 = x104*x56;
    const double x106 = x60*x70;
    const double x107 = x102*x62;
    const double x108 = x104*x64;
    const double x109 = x45*(x51*(-x46*x70 + x48*x70 + x50*x70) + x67*(-x59*(-2*x101 + 2*x103 + 2*x105) - x66*(-2*x106 + 2*x107 + 2*x108)));
    const double x110 = W0z*x7;
    const double x111 = W0x*x110;
    const double x112 = W0y*x110;
    const double x113 = -x111*x23 - x112*x25 + x27;
    const double x114 = hz*x113;
    const double x115 = x112*x9;
    const double x116 = x111*x27 + x23;
    const double x117 = x116*x19;
    const double x118 = x113*x22;
    const double x119 = hy*x118;
    const double x120 = x111*x9;
    const double x121 = x112*x27;
    const double x122 = x19*(-x121 - x25);
    const double x123 = hx*x118;
    const double x124 = x114*x75 - x12*x69 + x20;
    const double x125 = x30*x82;
    const double x126 = x113*x75;
    const double x127 = hx*x126 - x111*x20 + x22*(x121 + x25);
    const double x128 = x42*x82;
    const double x129 = 1 - x2;
    const double x130 = hy*x126 - x112*x20 + x116*x129;
    const double x131 = x36*x82;
    const double x132 = -x124*x125 - x127*x128 - x130*x131;
    const double x133 = W0x*hx;
    const double x134 = x11*x133;
    const double x135 = W0y*hy;
    const double x136 = x11*x135;
    const double x137 = -x23*x26 + x24*x25;
    const double x138 = hz*x137;
    const double x139 = x19*x39;
    const double x140 = x137*x22;
    const double x141 = hy*x140;
    const double x142 = x19*x32;
    const double x143 = hx*x140;
    const double x144 = x138*x75 + x68*(x134 + x136);
    const double x145 = x137*x75;
    const double x146 = hy*x145 + x129*x39 + x60;
    const double x147 = hx*x145 - x22*x32 - x52;
    const double x148 = -x125*x144 - x128*x147 - x131*x146;
    const double x149 = -x24*x36 + x26*x42;
    const double x150 = -x10*x30 + x36*x72 + x42*x71;
    const double x151 = qop0*s;
    const double x152 = x151*x52;
    const double x153 = x151*x2;
    const double x154 = x153*x33;
    const double x155 = x151*x19;
    const double x156 = x155*x56;
    const double x157 = x151*x60;
    const double x158 = x153*x62;
    const double x159 = x155*x64;
    const double x160 = x81/std::pow(B, 2);
    const double x161 = x160*x76;
    const double x162 = x151 - x153;
    const double x163 = x82*(qop0*x74 + x13*x151 - x151*x21 + x162*x49);
    const double x164 = x11*x160;
    const double x165 = x164*x88;
    const double x166 = qop0*x86 + x10*(x151*x31 + x151*x34 + x162*x56);
    const double x167 = x166*x92;
    const double x168 = x164*x96;
    const double x169 = -qop0*x94 + x10*(x151*x37 - x151*x41 + x162*x64);
    const double x170 = x169*x92;
    const double x171 = -x30*(-x161 + x163) - x36*(-x165 + x167) - x42*(-x168 + x170);
    const double x172 = x42*x44;
    const double x173 = -x31;
    const double x174 = x44*(x173 - x34 + x35);
    const double x175 = x172*(-x101 + x103 + x105) + x174*(-x106 + x107 + x108);
    const double x176 = x22*(W0z*hz + x133 + x135);
    const double x177 = W0y*x2 - hy*x176 + x19*(-W0x*hz + W0z*hx);
    const double x178 = x3 + x6;
    const double x179 = 1.0/x178;
    const double x180 = W0x*x2 - hx*x176 + x19*(W0y*hz - W0z*hy);
    const double x181 = x179*std::pow(x180, 2);
    const double x182 = std::pow(x177, 2)*x179;
    const double x183 = std::pow(x181 + x182, -1.0/2.0);
    const double x184 = x183/std::sqrt(x178*x7);
    const double x185 = x184*x7;
    const double x186 = x177*x185;
    const double x187 = x186*x96;
    const double x188 = x180*x185;
    const double x189 = x188*x88;
    const double x190 = x188*x82;
    const double x191 = x186*x82;
    const double x192 = -x102 + x70;
    const double x193 = x192*x56 + x31*x70 + x34*x70;
    const double x194 = x180*x184;
    const double x195 = x194*x92;
    const double x196 = x192*x64 + x37*x70 - x41*x70;
    const double x197 = x177*x184;
    const double x198 = x197*x92;
    const double x199 = x193*x195 - x196*x198;
    const double x200 = x179*x183*(W0z*x2 - hz*x176 + x19*(-x14 + x16));
    const double x201 = x180*x200;
    const double x202 = x177*x200;
    const double x203 = x181*x183 + x182*x183;
    const double x204 = x202*x82;
    const double x205 = x201*x82;
    const double x206 = x203*x82;
    const double x207 = -x193*x204 - x196*x205 + x206*(x13*x70 + x192*x49 - x21*x70);
    const double dqopdqop0 = 1;
    const double dqopdlam0 = 0;
    const double dqopdphi0 = 0;
    const double dqopdxt0 = 0;
    const double dqopdyt0 = 0;
    const double dqopdB = 0;
    const double dlamdqop0 = x100*x109 + x45*(x51*(-x0*x46 + x0*x48 + x0*x50) + x67*(-x59*(-2*x53 + 2*x55 + 2*x58) - x66*(-2*x61 + 2*x63 + 2*x65)));
    const double dlamdlam0 = x109*x132 + x45*(x51*(-x114*x22 - x46*x47 + x9) + x67*(-x59*(-2*x115 + 2*x117 - 2*x119) - x66*(-2*x120 + 2*x122 - 2*x123)));
    const double dlamdphi0 = x109*x148 + x45*(x51*(-x138*x22 + x20*(-x134 - x136)) + x67*(-x59*(2*x139 - 2*x141 + 2*x37) - x66*(2*x142 - 2*x143 - 2*x31)));
    const double dlamdxt0 = x109*x149;
    const double dlamdyt0 = x109*x150;
    const double dlamdB = x109*x171 + x45*(x51*(-x151*x46 + x151*x48 + x151*x50) + x67*(-x59*(-2*x152 + 2*x154 + 2*x156) - x66*(-2*x157 + 2*x158 + 2*x159)));
    const double dphidqop0 = x100*x175 + x172*(-x53 + x55 + x58) + x174*(-x61 + x63 + x65);
    const double dphidlam0 = x132*x175 + x172*(-x115 + x117 - x119) + x174*(-x120 + x122 - x123);
    const double dphidphi0 = x148*x175 + x172*(x139 - x141 + x37) + x174*(x142 - x143 + x173);
    const double dphidxt0 = x149*x175;
    const double dphidyt0 = x150*x175;
    const double dphidB = x171*x175 + x172*(-x152 + x154 + x156) + x174*(-x157 + x158 + x159);
    const double dxtdqop0 = x100*x199 + x187*x78 - x189*x78 + x190*x91 - x191*x98;
    const double dxtdlam0 = -x127*x198 + x130*x195 + x132*x199;
    const double dxtdphi0 = x146*x195 - x147*x198 + x148*x199;
    const double dxtdxt0 = W0x*x188 + W0y*x186 + x149*x199;
    const double dxtdyt0 = x111*x197 - x112*x194 + x150*x199;
    const double dxtdB = x160*x187 - x160*x189 + x166*x190 - x169*x191 + x171*x199;
    const double dytdqop0 = x100*x207 + x201*x97 - x201*x99 + x202*x90 - x202*x93 - x203*x79 + x203*x83;
    const double dytdlam0 = x124*x206 - x127*x205 - x130*x204 + x132*x207;
    const double dytdphi0 = x144*x206 - x146*x204 - x147*x205 + x148*x207;
    const double dytdxt0 = x149*x207 + x201*x26 - x202*x24;
    const double dytdyt0 = x10*x203 + x150*x207 + x201*x71 + x202*x72;
    const double dytdB = -x161*x203 + x163*x203 + x165*x202 - x167*x202 + x168*x201 - x170*x201 + x171*x207;
    
    res(0,0) = dqopdqop0;
    res(0,1) = dqopdlam0;
    res(0,2) = dqopdphi0;
    res(0,3) = dqopdxt0;
    res(0,4) = dqopdyt0;
    res(0,5) = dqopdB;
    res(1,0) = dlamdqop0;
    res(1,1) = dlamdlam0;
    res(1,2) = dlamdphi0;
    res(1,3) = dlamdxt0;
    res(1,4) = dlamdyt0;
    res(1,5) = dlamdB;
    res(2,0) = dphidqop0;
    res(2,1) = dphidlam0;
    res(2,2) = dphidphi0;
    res(2,3) = dphidxt0;
    res(2,4) = dphidyt0;
    res(2,5) = dphidB;
    res(3,0) = dxtdqop0;
    res(3,1) = dxtdlam0;
    res(3,2) = dxtdphi0;
    res(3,3) = dxtdxt0;
    res(3,4) = dxtdyt0;
    res(3,5) = dxtdB;
    res(4,0) = dytdqop0;
    res(4,1) = dytdlam0;
    res(4,2) = dytdphi0;
    res(4,3) = dytdxt0;
    res(4,4) = dytdyt0;
    res(4,5) = dytdB;
  }

  // convert to tesla for B field gradient
  res.col(5) *= 2.99792458e-3;
  
  return res;
                                                               
}

Matrix<double, 5, 6> ResidualGlobalCorrectionMaker::localTransportJacobian(const TrajectoryStateOnSurface &start,
                                            const std::pair<TrajectoryStateOnSurface, double> &propresult,
                                            bool doReverse) const {
  
  const TrajectoryStateOnSurface& state0 = doReverse ? propresult.first : start;
  const TrajectoryStateOnSurface& state1 = doReverse ? start : propresult.first;
  const GlobalVector& h = start.globalParameters().magneticFieldInInverseGeV();
  const double s = doReverse ? -propresult.second : propresult.second;
  
  // compute local to curvilinear jacobian at source
  JacobianLocalToCurvilinear local2curv(state0.surface(), state0.localParameters(), *state0.magneticField());
  const AlgebraicMatrix55& local2curvjac = local2curv.jacobian();
  const Matrix<double, 5, 5> H0 = Map<const Matrix<double, 5, 5, RowMajor>>(local2curvjac.Array());
  
  // compute curvilinear to local jacobian at destination
  JacobianCurvilinearToLocal curv2local(state1.surface(), state1.localParameters(), *state1.magneticField());
  const AlgebraicMatrix55& curv2localjac = curv2local.jacobian();
  const Matrix<double, 5, 5> H1 = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjac.Array());
  
//   // compute transport jacobian wrt curvlinear parameters
//   AnalyticalCurvilinearJacobian curv2curv;
//   curv2curv.computeFullJacobian(state0.globalParameters(), state1.globalParameters().position(), state1.globalParameters().momentum(), h, s);
//   const AlgebraicMatrix55& curv2curvjac = curv2curv.jacobian();
//   const Matrix<double, 5, 5> F = Map<const Matrix<double, 5, 5, RowMajor>>(curv2curvjac.Array());
//   
//   std::cout << "qbp" << std::endl;
//   std::cout << state0.localParameters().qbp() << std::endl;
//   
//   std::cout << "F" << std::endl;
//   std::cout << F << std::endl;
//   
//   // compute transport jacobian wrt B field (magnitude)
//   const Matrix<double, 5, 1> dF = bfieldJacobian(state0.globalParameters(), state1.globalParameters(), s, h);
//   
//   std::cout << "dF" << std::endl;
//   std::cout << dF << std::endl;
//   
//   Matrix<double, 5, 1> dFalt;
//   dFalt[0] = 0.;
//   dFalt.tail<4>() = 2.99792458e-3*state0.localParameters().qbp()/h.mag()*F.block<4, 1>(1, 0);
//   
//   std::cout << "dFalt" << std::endl;
//   std::cout << dFalt << std::endl;
//   
//   const Matrix<double, 5, 6> FdFfull = curvtransportgrad(state0.globalParameters(), state1.globalParameters(), s, h);
//   std::cout << "FdFfull" << std::endl;
//   std::cout << FdFfull << std::endl;  
//   
//   const Matrix<double, 5, 5> FmIcmssw = F - Matrix<double, 5, 5>::Identity();
//   std::cout << "FmIcmssw" << std::endl;
//   std::cout << FmIcmssw << std::endl;  
//   
//   const Matrix<double, 5, 5> FmI = FdFfull.block<5,5>(0,0) - Matrix<double, 5, 5>::Identity();
//   std::cout << "FmI" << std::endl;
//   std::cout << FmI << std::endl;  
  
  const Matrix<double, 5, 6> FdF = curvtransportJacobian(state0.globalParameters(), state1.globalParameters(), s, h);
  const Matrix<double, 5, 5> F = FdF.topLeftCorner<5,5>();
  const Matrix<double, 5, 1> dF = FdF.col(5);
  
  Matrix<double, 5, 6> res;
  res.leftCols<5>() = H1*F*H0;
  res.rightCols<1>() = H1*dF;
  return res;
                                              
}

Matrix<double, 5, 6> ResidualGlobalCorrectionMaker::curv2localTransportJacobian(const FreeTrajectoryState& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult,
                                              bool doReverse) const {
        
  const FreeTrajectoryState& end = *propresult.first.freeState();
  const TrajectoryStateOnSurface& proptsos = propresult.first;
  
  const FreeTrajectoryState& state0 = doReverse ? end : start;
  const FreeTrajectoryState& state1 = doReverse ? start : end;
  const GlobalVector& h = start.parameters().magneticFieldInInverseGeV();
  const double s = doReverse ? -propresult.second : propresult.second;
  
//   // compute transport jacobian wrt curvlinear parameters
//   AnalyticalCurvilinearJacobian curv2curv;
//   curv2curv.computeFullJacobian(state0.parameters(), state1.parameters().position(), state1.parameters().momentum(), h, s);
//   const AlgebraicMatrix55& curv2curvjac = curv2curv.jacobian();
//   const Matrix<double, 5, 5> F = Map<const Matrix<double, 5, 5, RowMajor>>(curv2curvjac.Array());
//   
//   // compute transport jacobian wrt B field (magnitude)
//   const Matrix<double, 5, 1> dF = bfieldJacobian(state0.parameters(), state1.parameters(), s, h);
  
  const Matrix<double, 5, 6> FdF = curvtransportJacobian(state0.parameters(), state1.parameters(), s, h);
  const Matrix<double, 5, 5> F = FdF.topLeftCorner<5,5>();
  const Matrix<double, 5, 1> dF = FdF.col(5);
  
  Matrix<double, 5, 6> res;
  
  if (doReverse) {
    // compute local to curvilinear jacobian at source
    JacobianLocalToCurvilinear local2curv(proptsos.surface(), proptsos.localParameters(), *proptsos.magneticField());
    const AlgebraicMatrix55& local2curvjac = local2curv.jacobian();
    const Matrix<double, 5, 5> H0 = Map<const Matrix<double, 5, 5, RowMajor>>(local2curvjac.Array());
    
    res.leftCols<5>() = F*H0;
    res.rightCols<1>() = dF;
  }
  else {
    // compute curvilinear to local jacobian at destination
    JacobianCurvilinearToLocal curv2local(proptsos.surface(), proptsos.localParameters(), *proptsos.magneticField());
    const AlgebraicMatrix55& curv2localjac = curv2local.jacobian();
    const Matrix<double, 5, 5> H1 = Map<const Matrix<double, 5, 5, RowMajor>>(curv2localjac.Array());
    
    res.leftCols<5>() = H1*F;
    res.rightCols<1>() = H1*dF;
  }
  
  return res;
                                                
}

Matrix<double, 5, 6> ResidualGlobalCorrectionMaker::curv2curvTransportJacobian(const FreeTrajectoryState& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult,
                                              bool doReverse) const {
        
  const FreeTrajectoryState& end = *propresult.first.freeState();
  const TrajectoryStateOnSurface& proptsos = propresult.first;
  
  const FreeTrajectoryState& state0 = doReverse ? end : start;
  const FreeTrajectoryState& state1 = doReverse ? start : end;
  const GlobalVector& h = start.parameters().magneticFieldInInverseGeV();
  const double s = doReverse ? -propresult.second : propresult.second;
  
//   // compute transport jacobian wrt curvlinear parameters
//   AnalyticalCurvilinearJacobian curv2curv;
//   curv2curv.computeFullJacobian(state0.parameters(), state1.parameters().position(), state1.parameters().momentum(), h, s);
//   const AlgebraicMatrix55& curv2curvjac = curv2curv.jacobian();
//   const Matrix<double, 5, 5> F = Map<const Matrix<double, 5, 5, RowMajor>>(curv2curvjac.Array());
//   
//   // compute transport jacobian wrt B field (magnitude)
//   const Matrix<double, 5, 1> dF = bfieldJacobian(state0.parameters(), state1.parameters(), s, h);
  
  const Matrix<double, 5, 6> FdF = curvtransportJacobian(state0.parameters(), state1.parameters(), s, h);

  return FdF;
}

AlgebraicVector5 ResidualGlobalCorrectionMaker::localMSConvolution(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const {
  
  const LocalVector lx(1.,0.,0.);
  const LocalVector ly(0.,1.,0.);
  const GlobalVector J = tsos.surface().toGlobal(lx);
  const GlobalVector K = tsos.surface().toGlobal(ly);
    
  const double Jx = J.x();
  const double Jy = J.y();
  const double Jz = J.z();
  
  const double Kx = K.x();
  const double Ky = K.y();
  const double Kz = K.z();
  
  const Matrix<double, 3, 1> p0(tsos.globalMomentum().x(), tsos.globalMomentum().y(), tsos.globalMomentum().z());
  const Matrix<double, 3, 1> W0 = p0.normalized();
  const Matrix<double, 3, 1> zhat(0., 0., 1.);
  
  const Matrix<double, 3, 1> U0 = zhat.cross(W0).normalized();
  const Matrix<double, 3, 1> V0 = W0.cross(U0);
    
  const double Ux = U0[0];
  const double Uy = U0[1];
  
  const double Vx = V0[0];
  const double Vy = V0[1];
  
  const double Wx = W0.x();
  const double Wy = W0.y();
  const double Wz = W0.z();
  
  const double x0 = std::pow(Wx, 2) + std::pow(Wy, 2) + std::pow(Wz, 2);
  const double x1 = std::pow(x0, -1.0/2.0);
  const double x2 = Ux*Wx + Uy*Wy;
  const double x3 = 1.0/x0;
  const double x4 = 2*x3;
  const double x5 = x2*x4;
  const double x6 = 3*x3;
  const double x7 = std::pow(Ux, 2) + std::pow(Uy, 2) - std::pow(x2, 2)*x6;
  const double x8 = x3*(Jx*Wx + Jy*Wy + Jz*Wz);
  const double x9 = Vx*Wx + Vy*Wy;
  const double x10 = x4*x9;
  const double x11 = std::pow(Vx, 2) + std::pow(Vy, 2) - x6*std::pow(x9, 2);
  const double x12 = x3*(Kx*Wx + Ky*Wy + Kz*Wz);
  const double d2dxdzdthetau2 = x1*(-x5*(Jx*Ux + Jy*Uy) - x7*x8);
  const double d2dxdzdthetav2 = x1*(-x10*(Jx*Vx + Jy*Vy) - x11*x8);
  const double d2dydzdthetau2 = x1*(-x12*x7 - x5*(Kx*Ux + Ky*Uy));
  const double d2dydzdthetav2 = x1*(-x10*(Kx*Vx + Ky*Vy) - x11*x12);

  
  
  const Surface& surface = tsos.surface();
  //
  //
  // Now get information on medium
  //
  const MediumProperties& mp = surface.mediumProperties();

  // Momentum vector
  LocalVector d = tsos.localMomentum();
  float p2 = d.mag2();
  d *= 1.f / sqrt(p2);
  float xf = 1.f / std::abs(d.z());  // increase of path due to angle of incidence
  // calculate general physics things
  constexpr float amscon = 1.8496e-4;  // (13.6MeV)**2
  const float m2 = updator.mass() * updator.mass();    // use mass hypothesis from constructor
  float e2 = p2 + m2;
  float beta2 = p2 / e2;
  // calculate the multiple scattering angle
  float radLen = mp.radLen() * xf;  // effective rad. length
  float sigt2 = 0.;                 // sigma(alpha)**2

  // Calculated rms scattering angle squared.
  float fact = 1.f + 0.038f * unsafe_logf<2>(radLen);
  fact *= fact;
  float a = fact / (beta2 * p2);
  sigt2 = amscon * radLen * a;
  
  const double sigma2 = sigt2;
  
  AlgebraicVector5 res;
  res[0] = 0.;
  res[1] = 0.5*d2dxdzdthetau2*sigma2 + 0.5*d2dxdzdthetav2*sigma2;
  res[2] = 0.5*d2dydzdthetau2*sigma2 + 0.5*d2dydzdthetav2*sigma2;
  res[3] = 0.;
  res[4] = 0.;

  return res;
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
//   dQddxdz(0,0) = delosddxdz;
//   dQddxdz(1,1) = dmsxxddxdz;
//   dQddxdz(1,2) = dmsxyddxdz;
//   dQddxdz(2,1) = dmsxyddxdz;
//   dQddxdz(2,2) = dmsyyddxdz;
  
  Matrix<double, 5, 5> &dQddydz = res[2];
  dQddydz = Matrix<double, 5, 5>::Zero();
//   dQddydz(0,0) = delosddydz;
//   dQddydz(1,1) = dmsxxddydz;
//   dQddydz(1,2) = dmsxyddydz;
//   dQddydz(2,1) = dmsxyddydz;
//   dQddydz(2,2) = dmsyyddydz;
  
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
  
  return res;
}

Matrix<double, 2, 1> ResidualGlobalCorrectionMaker::localPositionConvolution(const TrajectoryStateOnSurface& tsos) const {
  
  // curvilinear parameters
  const CurvilinearTrajectoryParameters curv(tsos.globalPosition(), tsos.globalMomentum(), tsos.charge());
  const double qop = curv.Qbp();
  const double lam = curv.lambda();
  const double phi = curv.phi();
//   const double xt = curv.xT();
//   const double yt = curv.yT();
  const double xt = 0.;
  const double yt = 0.;
  
  const Matrix<double, 3, 1> p0(tsos.globalMomentum().x(), tsos.globalMomentum().y(), tsos.globalMomentum().z());
  const Matrix<double, 3, 1> W0 = p0.normalized();
  const Matrix<double, 3, 1> zhat(0., 0., 1.);
  
  const Matrix<double, 3, 1> U0 = zhat.cross(W0).normalized();
  const Matrix<double, 3, 1> V0 = W0.cross(U0);
  
//   const Matrix<double, 3, 1> x0alt = xt*U0 + yt*V0;
//   std::cout << "global pos" << std::endl;
//   std::cout << tsos.globalPosition() << std::endl;
//   std::cout << "x0alt" << std::endl;
//   std::cout << x0alt << std::endl;
//   std::cout << "xt" << std::endl;
//   std::cout << xt << std::endl;
//   std::cout << "yt" << std::endl;
//   std::cout << yt << std::endl;
  
  const LocalVector lx(1.,0.,0.);
  const LocalVector ly(0.,1.,0.);
  const LocalVector lz(0.,0.,1.);
  const GlobalVector I = tsos.surface().toGlobal(lz);
  const GlobalVector J = tsos.surface().toGlobal(lx);
  const GlobalVector K = tsos.surface().toGlobal(ly);
  
  const LocalPoint l0(0., 0.);
  const GlobalPoint r = tsos.surface().toGlobal(l0);
  
  const double Ux = U0[0];
  const double Uy = U0[1];
//   const double Uz = U0[2];
  
  const double Vx = V0[0];
  const double Vy = V0[1];
//   const double Vz = V0[2];
  
  const double Ix = I.x();
  const double Iy = I.y();
  const double Iz = I.z();
  
  const double Jx = J.x();
  const double Jy = J.y();
  const double Jz = J.z();
  
  const double Kx = K.x();
  const double Ky = K.y();
  const double Kz = K.z();
  
  const double rx = r.x();
  const double ry = r.y();
  const double rz = r.z();
  
  const double pos0x = tsos.globalPosition().x();
  const double pos0y = tsos.globalPosition().y();
  const double pos0z = tsos.globalPosition().z();
  
  //sympy stuff goes here
  const double x0 = std::sin(lam);
  const double x1 = Iz*x0;
  const double x2 = std::cos(lam);
  const double x3 = std::cos(phi);
  const double x4 = Ix*x3;
  const double x5 = x2*x4;
  const double x6 = std::sin(phi);
  const double x7 = Iy*x6;
  const double x8 = x2*x7;
  const double x9 = x5 + x8;
  const double x10 = x1 + x9;
  const double x11 = 1.0/x10;
  const double x12 = Ix*rx;
  const double x13 = Iy*ry;
  const double x14 = Iz*rz;
  const double x15 = Ix*pos0x;
  const double x16 = Iy*pos0y;
  const double x17 = Iz*pos0z;
  const double x18 = Ux*xt;
  const double x19 = Ix*x18;
  const double x20 = Vx*yt;
  const double x21 = Ix*x20;
  const double x22 = Uy*xt;
  const double x23 = Iy*x22;
  const double x24 = Vy*yt;
  const double x25 = Iy*x24;
  const double x26 = x12 + x13 + x14 - x15 - x16 - x17 - x19 - x21 - x23 - x25;
  const double x27 = Iz*x2;
  const double x28 = x0*x4;
  const double x29 = x0*x7;
  const double x30 = x27 - x28 - x29;
  const double x31 = x2*x26;
  const double x32 = pos0z*x30 + x31;
  const double x33 = Jz*x11;
  const double x34 = pos0x + x18 + x20;
  const double x35 = x0*x26;
  const double x36 = -x3*x35 + x30*x34;
  const double x37 = Jx*x11;
  const double x38 = pos0y + x22 + x24;
  const double x39 = x30*x38 - x35*x6;
  const double x40 = Jy*x11;
  const double x41 = std::pow(x10, -2);
  const double x42 = -x27 + x28 + x29;
  const double x43 = x41*x42;
  const double x44 = pos0z*x10 + x35;
  const double x45 = Jz*x44;
  const double x46 = x3*x31;
  const double x47 = x10*x34 + x46;
  const double x48 = Jx*x47;
  const double x49 = x31*x6;
  const double x50 = x10*x38 + x49;
  const double x51 = Jy*x43;
  const double x52 = -x5 - x8;
  const double x53 = -x1 + x52;
  const double x54 = pos0z*x53 - x35;
  const double x55 = -x12 - x13 - x14 + x15 + x16 + x17 + x19 + x21 + x23 + x25;
  const double x56 = x2*x55;
  const double x57 = x3*x56;
  const double x58 = x34*x53 + x57;
  const double x59 = x38*x53 + x56*x6;
  const double x60 = Jz*x43;
  const double x61 = 2*x32;
  const double x62 = Jx*x43;
  const double x63 = 2*x36;
  const double x64 = 2*x39;
  const double x65 = std::pow(x10, -3);
  const double x66 = x42*x65;
  const double x67 = x66*(-2*x27 + 2*x28 + 2*x29);
  const double x68 = Jy*x50;
  const double x69 = x0*x6;
  const double x70 = Ix*x69;
  const double x71 = x0*x3;
  const double x72 = Iy*x71;
  const double x73 = x70 - x72;
  const double x74 = pos0z*x33;
  const double x75 = x2*x3;
  const double x76 = Iy*x75;
  const double x77 = x2*x6;
  const double x78 = Ix*x77;
  const double x79 = x76 - x78;
  const double x80 = pos0z*x79;
  const double x81 = x38*x73 + x55*x71;
  const double x82 = x34*x73 - x55*x69;
  const double x83 = x41*(-x70 + x72);
  const double x84 = -x76 + x78;
  const double x85 = x41*x84;
  const double x86 = x32*x85;
  const double x87 = x38*x79 + x46;
  const double x88 = -x49;
  const double x89 = x34*x79 + x88;
  const double x90 = Jx*x85;
  const double x91 = Jy*x85;
  const double x92 = -2*x76 + 2*x78;
  const double x93 = x66*x92;
  const double x94 = Ix*Ux;
  const double x95 = Iy*Uy;
  const double x96 = -x94 - x95;
  const double x97 = x2*x33;
  const double x98 = x0*x60;
  const double x99 = x94 + x95;
  const double x100 = Ux*x30 + x71*x99;
  const double x101 = Uy*x30 + x69*x99;
  const double x102 = x75*x96;
  const double x103 = Ux*x10 + x102;
  const double x104 = Uy*x10 + x77*x96;
  const double x105 = Ix*Vx;
  const double x106 = Iy*Vy;
  const double x107 = -x105 - x106;
  const double x108 = x105 + x106;
  const double x109 = Vx*x30 + x108*x71;
  const double x110 = Vy*x30 + x108*x69;
  const double x111 = x107*x75;
  const double x112 = Vx*x10 + x111;
  const double x113 = Vy*x10 + x107*x77;
  const double x114 = Jz*x85;
  const double x115 = 2*x80;
  const double x116 = x34*x52 + x57;
  const double x117 = x38*x52 + x88;
  const double x118 = x41*x9;
  const double x119 = 2*x87;
  const double x120 = 2*x89;
  const double x121 = x65*x84*x92;
  const double x122 = x0*x96;
  const double x123 = Ux*x79 + x77*x99;
  const double x124 = Uy*x79 + x102;
  const double x125 = x0*x107;
  const double x126 = Vx*x79 + x108*x77;
  const double x127 = Vy*x79 + x111;
  const double x128 = x0*x33;
  const double x129 = Kz*x11;
  const double x130 = Kx*x11;
  const double x131 = Ky*x11;
  const double x132 = Kz*x44;
  const double x133 = Kx*x43;
  const double x134 = Ky*x43;
  const double x135 = Kz*x43;
  const double x136 = Kx*x47;
  const double x137 = Ky*x50;
  const double x138 = pos0z*x129;
  const double x139 = Kx*x85;
  const double x140 = Ky*x85;
  const double x141 = x129*x2;
  const double x142 = Kz*x85;
  const double x143 = x0*x129;
  const double shat = x11*x26;
  const double dvdqop = 0;
  const double d2vdqopdqop = 0;
  const double d2vdqopdlam = 0;
  const double d2vdqopdphi = 0;
  const double d2vdqopdxt = 0;
  const double d2vdqopdyt = 0;
  const double dvdlam = x32*x33 + x36*x37 + x39*x40 + x43*x45 + x43*x48 + x50*x51;
  const double d2vdlamdlam = x33*x44 + x33*x54 + x37*x47 + x37*x58 + x40*x50 + x40*x59 + x45*x67 + x48*x67 + x51*x64 + x60*x61 + x62*x63 + x67*x68;
  const double d2vdlamdphi = Jz*x86 + x36*x90 + x37*x82 + x39*x91 + x40*x81 + x45*x83 + x45*x93 + x48*x83 + x48*x93 + x51*x87 + x60*x80 + x62*x89 + x68*x83 + x68*x93 + x73*x74;
  const double d2vdlamdxt = x100*x37 + x101*x40 + x103*x62 + x104*x51 + x96*x97 + x96*x98;
  const double d2vdlamdyt = x107*x97 + x107*x98 + x109*x37 + x110*x40 + x112*x62 + x113*x51;
  const double dvdphi = x37*x89 + x40*x87 + x45*x85 + x48*x85 + x50*x91 + x74*x79;
  const double d2vdphidphi = x114*x115 + x116*x37 + x117*x40 + x118*x45 + x118*x48 + x118*x68 + x119*x91 + x120*x90 + x121*x45 + x121*x48 + x121*x68 + x52*x74;
  const double d2vdphidxt = x103*x90 + x104*x91 + x114*x122 + x123*x37 + x124*x40;
  const double d2vdphidyt = x112*x90 + x113*x91 + x114*x125 + x126*x37 + x127*x40;
  const double dvdxt = x103*x37 + x104*x40 + x128*x96;
  const double d2vdxtdxt = 0;
  const double d2vdxtdyt = 0;
  const double dvdyt = x107*x128 + x112*x37 + x113*x40;
  const double d2vdytdyt = 0;
  const double dwdqop = 0;
  const double d2wdqopdqop = 0;
  const double d2wdqopdlam = 0;
  const double d2wdqopdphi = 0;
  const double d2wdqopdxt = 0;
  const double d2wdqopdyt = 0;
  const double dwdlam = x129*x32 + x130*x36 + x131*x39 + x132*x43 + x133*x47 + x134*x50;
  const double d2wdlamdlam = x129*x44 + x129*x54 + x130*x47 + x130*x58 + x131*x50 + x131*x59 + x132*x67 + x133*x63 + x134*x64 + x135*x61 + x136*x67 + x137*x67;
  const double d2wdlamdphi = Kz*x86 + x130*x82 + x131*x81 + x132*x83 + x132*x93 + x133*x89 + x134*x87 + x135*x80 + x136*x83 + x136*x93 + x137*x83 + x137*x93 + x138*x73 + x139*x36 + x140*x39;
  const double d2wdlamdxt = x100*x130 + x101*x131 + x103*x133 + x104*x134 + x122*x135 + x141*x96;
  const double d2wdlamdyt = x107*x141 + x109*x130 + x110*x131 + x112*x133 + x113*x134 + x125*x135;
  const double dwdphi = x130*x89 + x131*x87 + x132*x85 + x138*x79 + x139*x47 + x140*x50;
  const double d2wdphidphi = x115*x142 + x116*x130 + x117*x131 + x118*x132 + x118*x136 + x118*x137 + x119*x140 + x120*x139 + x121*x132 + x121*x136 + x121*x137 + x138*x52;
  const double d2wdphidxt = x103*x139 + x104*x140 + x122*x142 + x123*x130 + x124*x131;
  const double d2wdphidyt = x112*x139 + x113*x140 + x125*x142 + x126*x130 + x127*x131;
  const double dwdxt = x103*x130 + x104*x131 + x143*x96;
  const double d2wdxtdxt = 0;
  const double d2wdxtdyt = 0;
  const double dwdyt = x107*x143 + x112*x130 + x113*x131;
  const double d2wdytdyt = 0;
  Matrix<double, 5, 5> d2vdx2;
  d2vdx2(0, 0) = d2vdqopdqop;
  d2vdx2(0, 1) = d2vdqopdlam;
  d2vdx2(0, 2) = d2vdqopdphi;
  d2vdx2(0, 3) = d2vdqopdxt;
  d2vdx2(0, 4) = d2vdqopdyt;
  d2vdx2(1, 0) = d2vdqopdlam;
  d2vdx2(1, 1) = d2vdlamdlam;
  d2vdx2(1, 2) = d2vdlamdphi;
  d2vdx2(1, 3) = d2vdlamdxt;
  d2vdx2(1, 4) = d2vdlamdyt;
  d2vdx2(2, 0) = d2vdqopdphi;
  d2vdx2(2, 1) = d2vdlamdphi;
  d2vdx2(2, 2) = d2vdphidphi;
  d2vdx2(2, 3) = d2vdphidxt;
  d2vdx2(2, 4) = d2vdphidyt;
  d2vdx2(3, 0) = d2vdqopdxt;
  d2vdx2(3, 1) = d2vdlamdxt;
  d2vdx2(3, 2) = d2vdphidxt;
  d2vdx2(3, 3) = d2vdxtdxt;
  d2vdx2(3, 4) = d2vdxtdyt;
  d2vdx2(4, 0) = d2vdqopdyt;
  d2vdx2(4, 1) = d2vdlamdyt;
  d2vdx2(4, 2) = d2vdphidyt;
  d2vdx2(4, 3) = d2vdxtdyt;
  d2vdx2(4, 4) = d2vdytdyt;
  Matrix<double, 5, 5> d2wdx2;
  d2wdx2(0, 0) = d2wdqopdqop;
  d2wdx2(0, 1) = d2wdqopdlam;
  d2wdx2(0, 2) = d2wdqopdphi;
  d2wdx2(0, 3) = d2wdqopdxt;
  d2wdx2(0, 4) = d2wdqopdyt;
  d2wdx2(1, 0) = d2wdqopdlam;
  d2wdx2(1, 1) = d2wdlamdlam;
  d2wdx2(1, 2) = d2wdlamdphi;
  d2wdx2(1, 3) = d2wdlamdxt;
  d2wdx2(1, 4) = d2wdlamdyt;
  d2wdx2(2, 0) = d2wdqopdphi;
  d2wdx2(2, 1) = d2wdlamdphi;
  d2wdx2(2, 2) = d2wdphidphi;
  d2wdx2(2, 3) = d2wdphidxt;
  d2wdx2(2, 4) = d2wdphidyt;
  d2wdx2(3, 0) = d2wdqopdxt;
  d2wdx2(3, 1) = d2wdlamdxt;
  d2wdx2(3, 2) = d2wdphidxt;
  d2wdx2(3, 3) = d2wdxtdxt;
  d2wdx2(3, 4) = d2wdxtdyt;
  d2wdx2(4, 0) = d2wdqopdyt;
  d2wdx2(4, 1) = d2wdlamdyt;
  d2wdx2(4, 2) = d2wdphidyt;
  d2wdx2(4, 3) = d2wdxtdyt;
  d2wdx2(4, 4) = d2wdytdyt;
  Matrix<double, 5, 1> dvdx;
  dvdx[0] = dvdqop;
  dvdx[1] = dvdlam;
  dvdx[2] = dvdphi;
  dvdx[3] = dvdxt;
  dvdx[4] = dvdyt;
  Matrix<double, 5, 1> dwdx;
  dwdx[0] = dwdqop;
  dwdx[1] = dwdlam;
  dwdx[2] = dwdphi;
  dwdx[3] = dwdxt;
  dwdx[4] = dwdyt;


  std::cout << "dvdx" << std::endl;
  std::cout << dvdx << std::endl;
  
  std::cout << "dwdx" << std::endl;
  std::cout << dwdx << std::endl;
  
  std::cout << "d2vdx2" << std::endl;
  std::cout << d2vdx2 << std::endl;

  std::cout << "d2wdx2" << std::endl;
  std::cout << d2wdx2 << std::endl;
  
  std::cout << "shat" << std::endl;
  std::cout << shat << std::endl;
  
  // covariance matrix in curvilinear parameters
  const AlgebraicMatrix55 curvcovsmat = tsos.curvilinearError().matrix();
  
  // map to eigen data structure
  const Map<const Matrix<double, 5, 5, RowMajor>> curvcov(curvcovsmat.Array());
  
  // compute eigendecomposition
  SelfAdjointEigenSolver<Matrix<double, 5, 5>> es;
  es.compute(curvcov);
  
  // cov = VDV^(-1)
//   auto const& sqrtD = es.eigenvalues().cwiseSqrt();
  auto const& D = es.eigenvalues();
  auto const& V = es.eigenvectors();
  
  // compute second order correction to local positions
//   Matrix<double, 2, 1> res;
//   res[0] = 0.5*sqrtD.transpose()*V.transpose()*d2vdx2*V*sqrtD;
//   res[1] = 0.5*sqrtD.transpose()*V.transpose()*d2wdx2*V*sqrtD;
  
  Matrix<double, 2, 1> res = Matrix<double, 2, 1>::Zero();
  for (unsigned int i=0; i<5; ++i) {
    res[0] += 0.5*D[i]*V.col(i).transpose()*d2vdx2*V.col(i);
    res[1] += 0.5*D[i]*V.col(i).transpose()*d2wdx2*V.col(i);
  }
  
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
