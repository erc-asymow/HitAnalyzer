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

  TFile *fout;
  TTree *tree;

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


//   fout = new TFile("trackTreeGrads.root", "RECREATE");
//   fout = new TFile("trackTreeGradsdebug.root", "RECREATE");
  fout = new TFile("trackTreeGrads.root", "RECREATE");
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
//   tree->Branch("trackOrigParms", trackOrigParms.data(), "trackOrigParms[5]/F", basketSize);
//   tree->Branch("trackOrigCov", trackOrigCov.data(), "trackOrigCov[25]/F", basketSize);
  tree->Branch("trackParms", trackParms.data(), "trackParms[5]/F", basketSize);
  tree->Branch("trackCov", trackCov.data(), "trackCov[25]/F", basketSize);
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
    const unsigned int nparsEloss = nhits;
    const unsigned int npars = nparsAlignment + nparsBfield + nparsEloss;
    
    const unsigned int nstateparms = 5*(nhits+1);
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
    using MSScalar = AANT<double, 12>;;
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
    
    
    VectorXd gradfull = VectorXd::Zero(nparmsfull);
    MatrixXd hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);
    
//     VectorXd gradfull = chisq.value().derivatives();
//     MatrixXd hessfull = MatrixXd::Zero(nparmsfull, nparmsfull);
//     for (unsigned int i=0; i<nstateparms; ++i) {
//       hessfull.row(i) = chisq.derivatives()[i].derivatives();
//     }
    
    
    globalidxv.clear();
    globalidxv.resize(npars, 0);
    
    nParms = npars;
    tree->SetBranchAddress("globalidxv", globalidxv.data());
    
//     TrajectoryStateOnSurface currtsos;
    
    
    unsigned int parmidx = 0;

    
    auto const& refpoint = track.referencePoint();
    auto const& trackmom = track.momentum();
    
    const GlobalPoint refpos(refpoint.x(), refpoint.y(), refpoint.z());
    const GlobalVector refmom(trackmom.x(), trackmom.y(), trackmom.z());
    const GlobalTrajectoryParameters refglobal(refpos, refmom, track.charge(), field);
    const CurvilinearTrajectoryError referr(track.covariance());
    
    //initialize state
//     FreeTrajectoryState currentFts(refpos, refmom, track.charge(), field);
    FreeTrajectoryState currentFts(refglobal, referr);
    //inflate errors
    currentFts.rescaleError(100.);
    
    //propagate track from reference point to first measurement to compute the pathlength, and then compute
    //the transport jacobian corresponding to the reverse propagation
    
//     auto const& propresult = fPropagator.geometricalPropagator().propagateWithPath(refFts, innertsos.surface());
//     if (!propresult.first.isValid()) {
//       std::cout << "Abort: propagation from reference point failed" << std::endl;
//       continue;
//     }
//     AnalyticalCurvilinearJacobian curvjac(propresult.first.globalParameters(), refFts.position(), refFts.momentum(), -propresult.second);
//     const AlgebraicMatrix55& jacF = curvjac.jacobian();
//     Map<const Matrix<double, 5, 5, RowMajor> > Fref(jacF.Array());
    
    
    bool valid = true;
//     unsigned int ntotalhitdim = 0;
//     unsigned int alignmentidx = 0;
//     unsigned int bfieldidx = 0;
//     unsigned int elossidx = 0;
    
    for (unsigned int ihit = 0; ihit < track.recHitsSize(); ++ihit) {
      
      auto const& inhit = *(hitsbegin + ihit);
      //TODO check for null geographicalId?
      const GeomDet *detectorG = globalGeometry->idToDet(inhit->geographicalId());
      auto const& hit = inhit->cloneForFit(*detectorG);
//       const TransientTrackingRecHit hit = *inhit->cloneForFit(*detectorG);
      

      auto propresult = fPropagator->geometricalPropagator().propagateWithPath(currentFts, detectorG->surface());
      TrajectoryStateOnSurface& updtsos = propresult.first;
      double s = propresult.second;
      
      if (!updtsos.isValid()) {
        std::cout << "Abort: Propagation Failed!" << std::endl;
        valid = false;
        break;
      }
      
      //jacobian for propagated state (before material effects)
      JacobianCurvilinearToLocal hprop(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
      const AlgebraicMatrix55 &jachprop = hprop.jacobian();
      //efficient assignment from SMatrix using Eigen::Map
      Map<const Matrix<double, 5, 5, RowMajor> > jachpropeig(jachprop.Array());
      MSJacobian Hprop = jachpropeig.cast<MSScalar>();
      
      //compute transport jacobian
      AnalyticalCurvilinearJacobian curvjac(currentFts.parameters(), updtsos.globalParameters().position(), updtsos.globalParameters().momentum(), s);
      const AlgebraicMatrix55 &jacF = curvjac.jacobian();
      MSJacobian F = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array()).cast<MSScalar>();
      
      //bfield jacobian
      const MSVector dF = bfieldJacobian(currentFts.parameters(), updtsos.globalParameters(), s).cast<MSScalar>();
      
      //energy loss jacobian
      const Matrix<double, 5, 6> EdE = materialEffectsJacobian(updtsos, fPropagator->materialEffectsUpdator());
      const MSJacobian E = EdE.leftCols<5>().cast<MSScalar>();
      const MSVector dE = EdE.rightCols<1>().cast<MSScalar>();
      
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
      Map<const Matrix<double, 5, 5, RowMajor> >iQ(Qmat.Array());
      MSCovariance Qinv = MSCovariance::Zero();
      //Q is 3x3 in the upper left block because there is no displacement on thin scattering layers
      //so invert the upper 3x3 block
      Qinv.topLeftCorner<3,3>() = iQ.topLeftCorner<3,3>().inverse().cast<MSScalar>();
      
      //zero displacement on thin scattering layer approximated with small uncertainty
      const double epsxy = 1e-5; //0.1um
      Qinv(3,3) = MSScalar(2./epsxy/epsxy);
      Qinv(4,4) = MSScalar(2./epsxy/epsxy);
      
      //apply measurement update if applicable
      auto const& preciseHit = hit->isValid() ? cloner.makeShared(hit, updtsos) : hit;
      if (hit->isValid() && !preciseHit->isValid()) {
        std::cout << "Abort: Failed updating hit" << std::endl;
        valid = false;
        break;
      }
      //momentum kink residual
      //TODO rework code so this can be computed together with the update without loss of precision
      const AlgebraicVector5 idx0 = hit->isValid() ? update(updtsos, *preciseHit) : AlgebraicVector5(0., 0., 0., 0., 0.);
      updtsos = hit->isValid() ? updator.update(updtsos, *preciseHit) : updtsos;
      currentFts = *updtsos.freeState();
      
      MSVector dx0 = Map<const Vector5d>(idx0.Array()).cast<MSScalar>();
      
      //jacobian for updated state
      JacobianCurvilinearToLocal h(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
      const AlgebraicMatrix55 &jach = h.jacobian();
      //efficient assignment from SMatrix using Eigen::Map
      Map<const Matrix<double, 5, 5, RowMajor> > jacheig(jach.Array());
//       Hh.block<2,5>(2*i, 5*i) = jacheig.bottomRows<2>();
//       StateJacobian H = jacheig.cast<AAdouble>();
      
      if (true) {
        constexpr unsigned int nlocalstate = 10;
        constexpr unsigned int nlocalbfield = 1;
        constexpr unsigned int nlocaleloss = 1;
        constexpr unsigned int nlocalparms = nlocalbfield + nlocaleloss;
        
        constexpr unsigned int nlocal = nlocalstate + nlocalbfield + nlocaleloss;
        
        constexpr unsigned int localstateidx = 0;
        constexpr unsigned int localbfieldidx = localstateidx + nlocalstate;
        constexpr unsigned int localelossidx = localbfieldidx + nlocalbfield;
        constexpr unsigned int localparmidx = localbfieldidx;
        
        const unsigned int fullstateidx = 5*ihit;
        const unsigned int fullparmidx = nstateparms + parmidx;
        
        MSVector dxprev = MSVector::Zero();
        for (unsigned int j=0; j<dxprev.size(); ++j) {
          init_twice_active_var(dxprev[j], nlocal, localstateidx + j);
        }

        MSVector dx = MSVector::Zero();
        for (unsigned int j=0; j<dx.size(); ++j) {
          init_twice_active_var(dx[j], nlocal, localstateidx + 5 + j);
        }
        
        const MSJacobian H = jacheig.cast<MSScalar>();
        
        MSScalar dbeta(0.);
        init_twice_active_var(dbeta, nlocal, localbfieldidx);
        
        MSScalar dxi(0.);
        init_twice_active_var(dxi, nlocal, localelossidx);
        
        MSVector dms = dx0 + H*dx - E*Hprop*F*dxprev - E*Hprop*dF*dbeta - dE*dxi;
        MSScalar chisq = dms.transpose()*Qinv*dms;
        
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
      
      //hit information
      //FIXME consolidate this special cases into templated function(s)
      if (preciseHit->isValid()) {
        constexpr unsigned int nlocalstate = 2;
        constexpr unsigned int localstateidx = 0;
        constexpr unsigned int localalignmentidx = nlocalstate;
        constexpr unsigned int localparmidx = localalignmentidx;
        
        const unsigned int fullstateidx = 5*(ihit+1) + 3;
        const unsigned int fullparmidx = nstateparms+parmidx;

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
          
          const StripHit1DJacobian H = jacheig.block<1,2>(3,3).cast<StripHitScalar>();
          
          StripHitScalar dh = dy0 - (H*dx)[0] - A*dalpha;
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
          globalidxv[parmidx] = xglobalidx;
          parmidx++;
        }
        else if (preciseHit->dimension()==2) {
          bool ispixel = GeomDetEnumerators::isTrackerPixel(detectorG->subDetector());

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
            
            const PixelHit2DJacobian H = jacheig.bottomRightCorner<2,2>().cast<PixelHit2DScalar>();

            const PixelHit2DVector dh = dy0 - H*dx - A*dalpha;
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
              globalidxv[parmidx] = xglobalidx;
              parmidx++;
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
            
            const StripHit2DJacobian H = jacheig.bottomRightCorner<2,2>().cast<StripHitScalar>();

            StripHitVector dh = dy0 - H*dx - A*dalpha;
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
            globalidxv[parmidx] = xglobalidx;
            parmidx++;
            
          }
        }
      }
      
    }
    
    if (!valid) {
      continue;
    }
    
    //now do the expensive calculations and fill outputs
    auto const& dchisqdx = gradfull.head(nstateparms);
    auto const& dchisqdparms = gradfull.tail(npars);
    
    auto const& d2chisqdx2 = hessfull.topLeftCorner(nstateparms, nstateparms);
    auto const& d2chisqdxdparms = hessfull.topRightCorner(nstateparms, npars);
    auto const& d2chisqdparms2 = hessfull.bottomRightCorner(npars, npars);
    
    auto const& Cinvd = d2chisqdx2.ldlt();
    
    const Vector5d dxRef = -Cinvd.solve(dchisqdx).head<5>();
    const Matrix5d Cinner = Cinvd.solve(MatrixXd::Identity(nstateparms,nstateparms)).topLeftCorner<5,5>();
    
    const MatrixXd dxdparms = -Cinvd.solve(d2chisqdxdparms).transpose();
    
    const VectorXd grad = dchisqdparms + dxdparms*dchisqdx;
    const MatrixXd hess = d2chisqdparms2 + 2.*dxdparms*d2chisqdxdparms + dxdparms*d2chisqdx2*dxdparms.transpose();
    
    //fill output with corrected state and covariance at reference point
    refParms.fill(0.);
    refCov.fill(0.);
    const AlgebraicVector5& refVec = track.parameters();
    Map<Vector5f>(refParms.data()) = (Map<const Vector5d>(refVec.Array()) + dxRef).cast<float>();
    Map<Matrix<float, 5, 5, RowMajor> >(refCov.data()).triangularView<Upper>() = (2.*Cinner).cast<float>().triangularView<Upper>();
    
//     gradv.clear();
    jacrefv.clear();

//     gradv.resize(npars,0.);
    jacrefv.resize(5*npars, 0.);
    
    nJacRef = 5*npars;
//     tree->SetBranchAddress("gradv", gradv.data());
    tree->SetBranchAddress("jacrefv", jacrefv.data());
    
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
    
    
//     std::cout << "hess debug" << std::endl;
//     std::cout << "track parms" << std::endl;
//     std::cout << tkparms << std::endl;
//     std::cout << "dxRef" << std::endl;
//     std::cout << dxRef << std::endl;
//     std::cout << "original cov" << std::endl;
//     std::cout << track.covariance() << std::endl;
//     std::cout << "recomputed cov" << std::endl;
//     std::cout << 2.*Cinner << std::endl;

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
  
  TTree *gradtree = new TTree("gradtree","");
  double gradval;
  gradtree->Branch("gradval",&gradval);
  for (unsigned int i=0; i<gradagg.size(); ++i) {
    gradval = gradagg[i];
    gradtree->Fill();
  }
  
  TTree *hesstree = new TTree("hesstree","");
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
  
  fout->Write();
  fout->Close();
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
  
  runfout->Write();
  runfout->Close();
  
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
  double magb = b.norm();
  const Vector3d h = b.normalized();

  const Vector3d p0(globalSource.momentum().x(),
                      globalSource.momentum().y(),
                      globalSource.momentum().z());
  const Vector3d p1(globalDest.momentum().x(),
                      globalDest.momentum().y(),
                      globalDest.momentum().z());
  const Vector3d T0 = p0.normalized();
  double p = p1.norm();
  const Vector3d T = p1.normalized();
  double q = globalSource.charge();

  const Vector3d N0 = h.cross(T0).normalized();
  const double alpha = h.cross(T).norm();
  const double gamma = h.transpose()*T;

  //this is printed from sympy.printing.cxxcode together with sympy.cse for automatic substitution of common expressions
  auto const xf0 = q/p;
  auto const xf1 = s*xf0;
  auto const xf2 = magb*xf1;
  auto const xf3 = std::cos(xf2);
  auto const xf4 = T0;
  auto const xf5 = xf4;
  auto const xf6 = std::sin(xf2);
  auto const xf7 = alpha*xf6;
  auto const xf8 = N0;
  auto const xf9 = xf8;
  auto const xf10 = magb*xf0;
  auto const xf11 = xf10*xf3;
  auto const xf12 = 1.0/magb;
  auto const xf13 = p/q;
  auto const xf14 = gamma*xf12*xf13;
  auto const xf15 = h;
  auto const xf16 = xf15;
  auto const xf17 = 1 - xf3;
  auto const xf18 = xf3*(xf4.transpose()) + (-xf7)*xf8.transpose() + (gamma*xf17)*(xf15.transpose());
  auto const xf19 = s*xf12;
  auto const xf20 = xf13/std::pow(magb, 2);
  auto const xf21 = xf1*xf3;
  auto const xf22 = (xf19*xf3 - xf20*xf6)*xf5 + (alpha*xf17*xf20 - xf19*xf7)*xf9 + (gamma*xf20*(-xf2 + xf6) - xf14*(-xf1 + xf21))*xf16;
  auto const xf23 = xf1*xf6;
  auto const xf24 = xf10*xf6;
  auto const resf0 = -(xf3*xf5 + (-xf7)*xf9 + (-xf14*(-xf10 + xf11))*xf16)*xf18*xf22 + xf22;
  auto const resf1 = (-p)*((-xf24)*xf5 + (-alpha*xf11)*xf9 + (gamma*xf24)*xf16)*xf18*xf22 + p*(((-xf23)*xf5 + (-alpha*xf21)*xf9 + (gamma*xf23)*xf16));

  const Vector3d dMdB = resf0;
  const Vector3d dPdB = resf1;

  Vector6d dFglobal;
  dFglobal.head<3>() = dMdB;
  dFglobal.tail<3>() = dPdB;

  //convert to curvilinear
  JacobianCartesianToCurvilinear cart2curv(globalDest);
  const AlgebraicMatrix56& cart2curvjacs = cart2curv.jacobian();
  const Map<const Matrix<double, 5, 6, RowMajor> > cart2curvjac(cart2curvjacs.Array());

  //compute final jacobian (and convert to Tesla)
  Matrix<double, 5, 1> dF = 2.99792458e-3*cart2curvjac*dFglobal;
  //q/p element is exactly 0 by construction
  dF(0,0) = 0.;
  
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
