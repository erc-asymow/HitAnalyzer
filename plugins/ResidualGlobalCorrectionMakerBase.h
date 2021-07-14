#ifndef HitAnalyzer_ResidualGlobalCorrectionMakerBase_h
#define HitAnalyzer_ResidualGlobalCorrectionMakerBase_h


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
#include "CondFormats/Alignment/interface/Alignments.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/Track/interface/SimTrack.h"




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

class ResidualGlobalCorrectionMakerBase : public edm::stream::EDAnalyzer<>
{
public:
  explicit ResidualGlobalCorrectionMakerBase(const edm::ParameterSet &);
  ~ResidualGlobalCorrectionMakerBase();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  
  virtual void beginStream(edm::StreamID) override;
//   virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
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
                                                             
  Matrix<double, 5, 7> hybrid2curvTransportJacobian(const FreeTrajectoryState& start,
                                              const std::pair<TrajectoryStateOnSurface, double>& propresult) const;
                                              
  Matrix<double, 5, 7> hybrid2curvTransportJacobian(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s,
                                                             const GlobalVector& bfield) const;
                                                             
Matrix<double, 5, 7> hybrid2curvTransportJacobianVar(const GlobalTrajectoryParameters& globalSource,
                                                             const GlobalTrajectoryParameters& globalDest,
                                                             const double& s,
                                                             const GlobalVector& bfield) const;
                                                             
  AlgebraicVector5 localMSConvolution(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const;
                                                             
  Matrix<double, 5, 6> materialEffectsJacobian(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator);
  
  Matrix<double, 5, 6> materialEffectsJacobianVar(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator);
  
  std::array<Matrix<double, 5, 5>, 5> processNoiseJacobians(const TrajectoryStateOnSurface& tsos, const MaterialEffectsUpdator& updator) const;
  
  Matrix<double, 2, 1> localPositionConvolution(const TrajectoryStateOnSurface& tsos) const;
  
  template <unsigned int D>
  AlgebraicVector5 lupdate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  AlgebraicVector5 update(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit);

  template <typename T>
  void init_twice_active_var(T &ad, const unsigned int d_num, const unsigned int idx) const;
  
  template <typename T>
  void init_twice_active_null(T &ad, const unsigned int d_num) const;
  
//   Matrix<double, 5, 3> vertexToCurvilinearJacobian(const FreeTrajectoryState &state) const;
  Matrix<double, 6, 6> cartesianToCartesianJacobian(const FreeTrajectoryState &state) const;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;
  edm::EDGetTokenT<std::vector<int>> genParticlesBarcodeToken_;
//   edm::EDGetTokenT<TrajTrackAssociationCollection> inputTrack_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrack_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrackOrig_;
  edm::EDGetTokenT<std::vector<int> > inputIndices_;
  edm::EDGetTokenT<reco::BeamSpot> inputBs_;
//   edm::EDGetTokenT<std::vector<PSimHit>> inputSimHits_;
  std::vector<edm::EDGetTokenT<std::vector<PSimHit>>> inputSimHits_;
  edm::EDGetTokenT<std::vector<SimTrack>> inputSimTracks_;
  
  edm::EDGetTokenT<reco::MuonCollection> inputMuons_;

  std::string corFile_;
  
//   SiStripClusterInfo siStripClusterInfo_;

  
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
  
  float genX;
  float genY;
  float genZ;
  
  unsigned int nHits;
  unsigned int nValidHits;
  unsigned int nValidPixelHits;
  unsigned int nParms;
  unsigned int nJacRef;
  unsigned int nSym;
  
  unsigned int nValidHitsFinal;
  unsigned int nValidPixelHitsFinal;
  
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
  std::vector<float> dxerr;
  std::vector<float> dyerr;
  
  std::vector<int> clusterSize;
  std::vector<int> clusterSizeX;
  std::vector<int> clusterSizeY;
  std::vector<int> clusterCharge;
  
  std::vector<int> clusterChargeBin;
  std::vector<int> clusterOnEdge;
  
  std::vector<float> clusterProbXY;
  std::vector<float> clusterSN;
  
  std::vector<float> dxreccluster;
  std::vector<float> dyreccluster;
  
  
  std::vector<float> localqop;
  std::vector<float> localdxdz;
  std::vector<float> localdydz;
  std::vector<float> localx;
  std::vector<float> localy;
  
  std::map<std::pair<int, DetId>, unsigned int> detidparms;
  std::vector<std::pair<int, DetId>> detidparmsrev;
  
  std::map<DetId, ReferenceCountingPointer<Plane>> surfacemap_;
  std::vector<float> corparms_;
  
  unsigned int run;
  unsigned int lumi;
  unsigned long long event;
  
  std::vector<double> gradagg;
  
  std::unordered_map<std::pair<unsigned int, unsigned int>, double> hessaggsparse;
  
  bool fitFromGenParms_;
  bool fillTrackTree_;
  bool fillGrads_;
  
  bool debugprintout_;
  
  bool doGen_;
  bool doSim_;
  bool doMuons_;
  
  bool bsConstraint_;
  
  bool applyHitQuality_;
  
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
  float simtestvz;
  float simtestzlocalref;
  float simtestrho;
  float simtestdx;
  float simtestdxrec;
  float simtestdy;
  float simtestdyrec;
  float simtestdxprop;
  float simtestdyprop;
  unsigned int simtestdetid;
  
  std::vector<float> rx;
  std::vector<float> ry;
  
  std::vector<float> deigx;
  std::vector<float> deigy;
  
  float edmval;
  unsigned int niter;
  
  float chisqval;
  unsigned int ndof;
  
//   bool filledRunTree_;
  
};

template <unsigned int D>
AlgebraicVector5 ResidualGlobalCorrectionMakerBase::lupdate(const TrajectoryStateOnSurface& tsos, const TrackingRecHit& aRecHit) {
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

template <typename T>
void ResidualGlobalCorrectionMakerBase::init_twice_active_var(T &ad, const unsigned int d_num, const unsigned int idx) const {
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
void ResidualGlobalCorrectionMakerBase::init_twice_active_null(T &ad, const unsigned int d_num) const {
  // initialize derivative direction in value field of outer active variable
  ad.value().derivatives() = T::DerType::Scalar::DerType::Zero(d_num);
  // initialize derivatives direction of the variable
  ad.derivatives() = T::DerType::Zero(d_num);
  // initialize Hessian matrix of variable to zero
  for(unsigned int idx=0;idx<d_num;idx++){
    ad.derivatives()(idx).derivatives()  = T::DerType::Scalar::DerType::Zero(d_num);
  }
}

#endif
