// -*- C++ -*-
//
// Package:    TrackAnalysis/HitAnalyzer
// Class:      HitAnalyzer
//
/**\class HitAnalyzer HitAnalyzer.cc TrackAnalysis/HitAnalyzer/plugins/HitAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michail Bachtis
//         Created:  Mon, 21 Mar 2016 14:17:37 GMT
//
//

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
#include "../interface/OffsetMagneticField.h"


#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "functions.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

using namespace Eigen;

constexpr unsigned int max_n = 25; //!< In order to avoid use of dynamic memory

typedef Matrix<double, Dynamic, Dynamic, 0, max_n, max_n> MatrixNd;
typedef Array<double, Dynamic, Dynamic, 0, max_n, max_n> ArrayNd;
typedef Matrix<double, Dynamic, Dynamic, 0, 2 * max_n, 2 * max_n> Matrix2Nd;
typedef Matrix<double, Dynamic, Dynamic, 0, 3 * max_n, 3 * max_n> Matrix3Nd;
typedef Matrix<double, 2, Dynamic, 0, 2, max_n> Matrix2xNd;
typedef Array<double, 2, Dynamic, 0, 2, max_n> Array2xNd;
typedef Matrix<double, 3, Dynamic, 0, 3, max_n> Matrix3xNd;
typedef Matrix<double, Dynamic, 3, 0, max_n, 3> MatrixNx3d;
typedef Matrix<double, Dynamic, 5, 0, max_n, 5> MatrixNx5d;
typedef Matrix<double, Dynamic, 1, 0, max_n, 1> VectorNd;
typedef Matrix<double, Dynamic, 1, 0, 2 * max_n, 1> Vector2Nd;
typedef Matrix<double, Dynamic, 1, 0, 3 * max_n, 1> Vector3Nd;
typedef Matrix<double, 1, Dynamic, 1, 1, max_n> RowVectorNd;
typedef Matrix<double, 1, Dynamic, 1, 1, 2 * max_n> RowVector2Nd;
typedef Matrix<double, 2, 2> Matrix2d;
typedef Matrix<double, 5, 5> Matrix5d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 5, 6> Matrix56d;
typedef Matrix<double, 5, 1> Vector5d;
typedef Matrix<double, 6, 1> Vector6d;

typedef ROOT::Math::SMatrix<double, 2> SMatrix22;

//too big for stack :(
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 5*max_n> GlobalParameterMatrix;
// typedef Matrix<double, Dynamic, 1, 0, 5*max_n, 1> GlobalParameterVector;
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 2*max_n> AlignmentJacobianMatrix;
// typedef Matrix<double, Dynamic, Dynamic, 0, 5*max_n, 2*max_n> TransportJacobianMatrix;

typedef MatrixXd GlobalParameterMatrix;
typedef VectorXd GlobalParameterVector;
typedef MatrixXd AlignmentJacobianMatrix;
typedef MatrixXd TransportJacobianMatrix;
typedef MatrixXd ELossJacobianMatrix;


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

  // ----------member data ---------------------------
  //  edm::EDGetTokenT<reco::TrackCollection>      inputTracks_;
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;
  edm::EDGetTokenT<reco::TrackCollection> inputTrack_;

  TFile *fout;
  TTree *tree;

  const int N = 25;

  int n;
  std::vector<float> z;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> r;

  std::vector<float> xUnc;
  std::vector<float> yUnc;
  std::vector<float> zUnc;
  std::vector<float> etaUnc;
  std::vector<float> phiUnc;
  std::vector<float> rUnc;

  std::vector<float> pt;

  std::vector<int> detector;
  std::vector<int> stereo;
  std::vector<int> glued;
  std::vector<int> layer;

  //local positions and errors
  std::vector<float> localx;
  std::vector<float> localy;
  std::vector<float> localz;
  std::vector<float> localxErr;
  std::vector<float> localyErr;
  std::vector<float> localxyErr;

  std::vector<float> globalrErr;
  std::vector<float> globalzErr;
  std::vector<float> globalrphiErr;
  std::vector<float> localx_state;
  std::vector<float> localy_state;

  //material stuff
  std::vector<std::vector<double>> trackQ;
  std::vector<std::vector<double>> trackH;
  std::vector<std::vector<double>> trackHprop;
  std::vector<std::vector<double>> trackF;
  std::vector<std::vector<double>> trackC;

  std::vector<std::vector<double>> updState;
  std::vector<std::vector<double>> backPropState;
  std::vector<std::vector<double>> forwardPropState;

  std::vector<std::vector<double>> updStateLocal;
  std::vector<std::vector<double>> fwdPredStateLocal;
  std::vector<std::vector<double>> bkgPredStateLocal;
  
  std::vector<std::vector<double>> bkgupdPropState;
  
  std::vector<std::vector<double>> curvpars;
  std::vector<int> hitDimension;
  std::vector<unsigned int> detid;
  
  std::vector<double> gradv;
  std::vector<double> hessv;
  std::vector<unsigned int> globalidxv;
  
//   std::vector<std::pair<int, DetId> > detidparms;
  std::map<std::pair<int, DetId>, int> detidparms;

  float trackEta;
  float trackPhi;
  float trackPt;
  float trackPtErr;
  float trackZ0;
  float trackX0;
  float trackY0;
  float trackCharge;

  float genPt;
  float genCharge;
  int ninvalidHits;
  unsigned int nglobalparms;
  unsigned int nglobalparmsalignment;
  unsigned int nglobalparmsbfield;
  unsigned int nglobalparmseloss;
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
  //  inputTracks_ = consumes<reco::TrackCollection>(edm::InputTag("TrackRefitter"));
  inputTraj_ = consumes<std::vector<Trajectory>>(edm::InputTag("TrackRefitter"));
  GenParticlesToken_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));
  inputTrack_ = consumes<reco::TrackCollection>(edm::InputTag("TrackRefitter"));

  n = 0;

  fout = new TFile("trackTreeGrads.root", "RECREATE");
  tree = new TTree("tree", "tree");

  tree->Branch("n", &n, "n/I");
  tree->Branch("ninvalidHits", &ninvalidHits, "ninvalidHits/I");
  tree->Branch("z", &z);
  tree->Branch("eta", &eta);
  tree->Branch("phi", &phi);
  tree->Branch("r", &r);
  tree->Branch("pt", &pt);

  tree->Branch("xUnc", &xUnc);
  tree->Branch("yUnc", &yUnc);
  tree->Branch("zUnc", &zUnc);
  tree->Branch("etaUnc", &etaUnc);
  tree->Branch("phiUnc", &phiUnc);
  tree->Branch("rUnc", &rUnc);

  tree->Branch("stereo", &stereo);
  tree->Branch("glued", &glued);
  tree->Branch("detector", &detector);
  tree->Branch("layer", &layer);

  tree->Branch("trackPt", &trackPt, "trackPt/F");
  tree->Branch("trackPtErr", &trackPtErr, "trackPtErr/F");
  tree->Branch("trackEta", &trackEta, "trackEta/F");
  tree->Branch("trackPhi", &trackPhi, "trackPhi/F");
  tree->Branch("trackX0", &trackX0, "trackX0/F");
  tree->Branch("trackY0", &trackY0, "trackY0/F");
  tree->Branch("trackZ0", &trackZ0, "trackZ0/F");
  tree->Branch("trackCharge", &trackCharge, "trackCharge/F");

  tree->Branch("genPt", &genPt, "genPt/F");
  tree->Branch("genCharge", &genCharge, "genCharge/F");

  tree->Branch("localx", &localx);
  tree->Branch("localy", &localy);
  tree->Branch("localz", &localz);
  tree->Branch("localx_state", &localx_state);
  tree->Branch("localy_state", &localy_state);
  tree->Branch("localxErr", &localxErr);
  tree->Branch("localyErr", &localyErr);
  tree->Branch("localxyErr", &localxyErr);
  tree->Branch("globalrErr", &globalrErr);
  tree->Branch("globalzErr", &globalzErr);
  tree->Branch("globalrphiErr", &globalrphiErr);
  tree->Branch("trackQ", &trackQ);
  tree->Branch("trackH", &trackH);
  tree->Branch("trackHprop", &trackHprop);
  tree->Branch("trackF", &trackF);
  tree->Branch("trackC", &trackC);
  tree->Branch("updState", &updState);
  tree->Branch("backPropState", &backPropState);
  tree->Branch("forwardPropState", &forwardPropState);
  tree->Branch("updStateLocal", &updStateLocal);
  tree->Branch("fwdPredStateLocal", &fwdPredStateLocal);
  tree->Branch("bkgPredStateLocal", &bkgPredStateLocal);
  tree->Branch("bkgupdPropState", &bkgupdPropState);
  tree->Branch("curvpars", &curvpars);
  tree->Branch("hitDimension", &hitDimension);
  tree->Branch("detid", &detid);
  
  tree->Branch("gradv", &gradv);
  tree->Branch("hessv", &hessv);
  tree->Branch("globalidxv", &globalidxv);
  tree->Branch("nglobalparms",&nglobalparms);
  tree->Branch("nglobalparmsalignment",&nglobalparmsalignment);
  tree->Branch("nglobalparmsbfield",&nglobalparmsbfield);
  tree->Branch("nglobalparmseloss",&nglobalparmseloss);
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

  Handle<std::vector<Trajectory>> trajH;
  iEvent.getByToken(inputTraj_, trajH);
  
  Handle<reco::TrackCollection> trackH;
  iEvent.getByToken(inputTrack_, trackH);

  ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  auto field = magfield.product();
  
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrh;
  iSetup.get<TransientRecHitRecord>().get("WithAngleAndTemplate",ttrh);

//   OffsetMagneticField offsetField(field, GlobalVector(0.,0.,0.));
  
  printf("traj size = %i, track size = %i\n", int(trajH->size()), int(trackH->size()));
  
  for (unsigned int j = 0; j < trajH->size(); ++j)
  {
    printf("j = %i\n", int(j));

    const Trajectory &traj = (*trajH)[j];
    if (traj.isLooper()) {
      continue;
    }
    
    //     const reco::Track& track = (*trackH)[i];
    //     if (track.lost()>0)
    //       continue;

//     const reco::Track& track = (*trackH)[j];
//     trackPt = track.pt();
//     trackEta = track.eta();
//     trackPhi = track.phi();
//     trackCharge = track.charge();
//     trackPtErr = track.ptError();
    
    const std::vector<TrajectoryMeasurement> &tms = (*trajH)[j].measurements();

    
    ////

    if (((*trajH)[j].direction()) == alongMomentum)
    {

      TrajectoryStateOnSurface measurement = (*trajH)[j].firstMeasurement().updatedState();

      trackPt = measurement.globalMomentum().perp();
      //FIX BUG
      trackPtErr = sqrt(measurement.curvilinearError().matrix()[0][0]) * trackPt;

      trackEta = measurement.globalMomentum().eta();
      trackPhi = measurement.globalMomentum().phi();
      trackX0 = measurement.globalPosition().x();
      trackY0 = measurement.globalPosition().y();
      trackZ0 = measurement.globalPosition().z();
      trackCharge = measurement.charge();
    }
    else
    {

      TrajectoryStateOnSurface measurement = (*trajH)[j].lastMeasurement().updatedState();

      trackPt = measurement.globalMomentum().perp();
      trackPtErr = sqrt(measurement.curvilinearError().matrix()[0][0]) * trackPt;
      trackEta = measurement.globalMomentum().eta();
      trackPhi = measurement.globalMomentum().phi();
      trackX0 = measurement.globalPosition().x();
      trackY0 = measurement.globalPosition().y();
      trackZ0 = measurement.globalPosition().z();
      trackCharge = measurement.charge();
    }

    //     printf("First point %f %f %f  - %f %f %f\n",trackX0,trackY0,trackZ0,trackPt,trackEta,trackPhi);

    for (std::vector<reco::GenParticle>::const_iterator g = genParticles.begin(); g != genParticles.end(); ++g)
    {

      float dR = deltaR(g->phi(), trackPhi, g->eta(), trackEta);

      if (dR < 0.15)
      {
        genPt = g->pt();
        genCharge = g->charge();
      }
      else
        continue;
    }

    ////
    n = 0;
    ninvalidHits = 0;

    z.clear();
    eta.clear();
    phi.clear();
    r.clear();

    xUnc.clear();
    yUnc.clear();
    zUnc.clear();
    etaUnc.clear();
    phiUnc.clear();
    rUnc.clear();

    pt.clear();

    detector.clear();
    stereo.clear();
    glued.clear();
    layer.clear();

    //local positions and errors
    localx.clear();
    localy.clear();
    localz.clear();
    localxErr.clear();
    localyErr.clear();
    localxyErr.clear();

    localx_state.clear();
    localy_state.clear();

    globalrErr.clear();
    globalzErr.clear();
    globalrphiErr.clear();

    updState.clear();
    backPropState.clear();
    forwardPropState.clear();

    updStateLocal.clear();
    fwdPredStateLocal.clear();
    bkgPredStateLocal.clear();
    
    bkgupdPropState.clear();
    
    //material stuff
    trackQ.clear();
    trackH.clear();
    trackF.clear();
    trackC.clear();
    trackHprop.clear();
    
    hitDimension.clear();
    detid.clear();

//     const float mass = 0.1395703;
    const float mass = 0.105;
    const float maxDPhi = 1.6;
//     printf("traj propdir = %i\n", int((*trajH)[j].direction()));
    

    PropagationDirection rpropdir = (*trajH)[j].direction();
    PropagationDirection fpropdir = rpropdir == alongMomentum ? oppositeToMomentum : alongMomentum;
    
    //TODO properly handle the outside-in case
    assert(fpropdir == alongMomentum);
    
    
//     PropagatorWithMaterial rPropagator(rpropdir, mass, &offsetField, maxDPhi, true, -1., false);
    PropagatorWithMaterial rPropagator(rpropdir, mass, field, maxDPhi, true, -1., false);
//     PropagatorWithMaterial fPropagator(fpropdir, mass, field, maxDPhi, true, -1., false);
    
//     KFSwitching1DUpdator updator;
    KFUpdator updator;
    TrajectoryStateCombiner combiner;
    
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
    
//     VectorNd grad = VectorNd::Zero(npars);
//     MatrixNd hess = MatrixNd::Zero(npars,npars);
    
    //curvilinear to local jacobian (at two points)
    GlobalParameterMatrix H = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    GlobalParameterMatrix Hprop = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    //alignment jacobian
    AlignmentJacobianMatrix A = AlignmentJacobianMatrix::Zero(5*nhits, nparsAlignment);
    //propagation jacobian wrt parameters
    GlobalParameterMatrix F = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    //energy loss jacobian wrt parameters
    GlobalParameterMatrix E = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    //propagation jacobian with respect to b-field
    TransportJacobianMatrix dF = TransportJacobianMatrix::Zero(5*nhits, nparsBfield);
    //energy loss jacobian wrt energy loss
    ELossJacobianMatrix dE = GlobalParameterMatrix::Zero(5*nhits, nparsEloss);
//     //identity matrix
//     GlobalParameterMatrix const I = GlobalParameterMatrix::Identity(5*nhits, 5*nhits);
    
    //hit covariance matrix
    GlobalParameterMatrix Vinv = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    //process noise matrix (MS+stochastic energy loss)
    GlobalParameterMatrix Qinv = GlobalParameterMatrix::Zero(5*nhits, 5*nhits);
    
    //hit residuals
    GlobalParameterVector dy0 = GlobalParameterVector::Zero(5*nhits);
    //momentum kink residuals
    GlobalParameterVector dx0 = GlobalParameterVector::Zero(5*nhits);
    
    
    gradv.clear();
    hessv.clear();
    globalidxv.clear();
    
    globalidxv.resize(npars, 0);
    gradv.resize(npars,0.);
    hessv.resize(npars*npars, 0.);

    //eigen representation of the underlying vector storage
    Map<GlobalParameterVector> grad(gradv.data(), npars);
    Map<GlobalParameterMatrix> hess(hessv.data(), npars, npars);

    
    unsigned int alignmentidx = 0;
    unsigned int bfieldidx = 0;
    unsigned int elossidx = 0;
    for (unsigned int i = 0; i < tms.size(); ++i)
    {
      
      TrajectoryMeasurement const& tm = tms[i];
      auto const& hit = tm.recHit();
      const GeomDet *detectorG = globalGeometry->idToDet(hit->geographicalId());
      TrajectoryStateOnSurface const& updtsos = tm.updatedState();
      
      //TODO properly handle this case
      assert(updtsos.isValid());
      
      //hit information
      if (hit->isValid()) {
        //fill x residual
        dy0(5*i+3) = hit->localPosition().x() - updtsos.localPosition().x();
        
        //check for 2d hits
        if (hit->dimension()>1) {
          //compute 2x2 covariance matrix and invert
          Matrix2d iV;
          iV << hit->localPositionError().xx(), hit->localPositionError().xy(),
                hit->localPositionError().xy(), hit->localPositionError().yy();
          Vinv.block<2,2>(5*i+3, 5*i+3) = iV.inverse();
        }
        else {
          //covariance is 1x1
          Vinv(5*i+3, 5*i+3) = 1./hit->localPositionError().xx();
        }
      }
      
      JacobianCurvilinearToLocal h(updtsos.surface(), updtsos.localParameters(), *updtsos.magneticField());
      const AlgebraicMatrix55 &jach = h.jacobian();
      //efficient assignment from SMatrix using Eigen::Map
      H.block<5,5>(5*i, 5*i) = Map<const Matrix<double, 5, 5, RowMajor> >(jach.Array());
      
//       std::cout << "H:" << std::endl;
//       std::cout << H.block<5,5>(5*i, 5*i) << std::endl;
      
      //fill alignment jacobian
      bool ispixel = GeomDetEnumerators::isTrackerPixel(detectorG->subDetector());
      //2d alignment correction for pixel hits, otherwise 1d
      A(5*i+3, alignmentidx) = 1.;
//       const unsigned int xglobalidx = std::lower_bound(detidparms.begin(), detidparms.end(), std::make_pair(0,hit->geographicalId())) - detidparms.begin();
      const unsigned int xglobalidx = detidparms[std::make_pair(0,hit->geographicalId())];
//       assert(xglobalidx<detidparms.size());
      globalidxv[alignmentidx] = xglobalidx;
      alignmentidx++;
      if (ispixel) {
        A(5*i+4, alignmentidx) = 1.;
//         const unsigned int yglobalidx = std::lower_bound(detidparms.begin(), detidparms.end(), std::make_pair(1,hit->geographicalId())) - detidparms.begin();
        const unsigned int yglobalidx = detidparms[std::make_pair(1,hit->geographicalId())];
//         assert(yglobalidx<detidparms.size());
        globalidxv[alignmentidx] = yglobalidx;
        alignmentidx++;
      }

      
      if (i >0) {
        //TODO check consistency of propagation jacobians when propagating oppositeToMomentum!!!
        
        //compute transport jacobian propagating outside in to current layer
        TrajectoryStateOnSurface const& toproptsos = tms[i-1].updatedState();
        
        //no offset for nominal propagation
        const auto &propresult = rPropagator.geometricalPropagator().propagateWithPath(toproptsos, updtsos.surface());
        AnalyticalCurvilinearJacobian curvjac(toproptsos.globalParameters(), propresult.first.globalParameters().position(), propresult.first.globalParameters().momentum(), propresult.second);
        const AlgebraicMatrix55 &jacF = curvjac.jacobian();
        
        F.block<5,5>(5*i, 5*(i-1)) = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array());
        
        //analytic jacobian wrt magnitude of magnetic field
        //TODO should we parameterize with respect to z-component instead?
        //extending derivation from CMS NOTE 2006/001
        const AlgebraicVector3 b(toproptsos.globalParameters().magneticFieldInInverseGeV().x(),
                           toproptsos.globalParameters().magneticFieldInInverseGeV().y(),
                           toproptsos.globalParameters().magneticFieldInInverseGeV().z());
        double magb = ROOT::Math::Dot(b,b);
        const AlgebraicVector3 h = b/magb;

        const AlgebraicVector3 p0(toproptsos.globalParameters().momentum().x(),
                            toproptsos.globalParameters().momentum().y(),
                            toproptsos.globalParameters().momentum().z());
        double p = ROOT::Math::Dot(p0,p0);
        const AlgebraicVector3 T0 = p0/p;
        double q = toproptsos.charge();
        
        const AlgebraicVector3 hcrossT0 = ROOT::Math::Cross(h,T0);
        double alpha = ROOT::Math::Dot(hcrossT0,hcrossT0);
        const AlgebraicVector3 N0 = hcrossT0/alpha;
        double gamma = ROOT::Math::Dot(h,T0);
        double s = propresult.second;
        
        //code generated by sympy
        const AlgebraicVector3 dMdB = (s*std::cos(magb*q*s/p)/magb - p*std::sin(magb*q*s/p)/(std::pow(magb, 2)*q))*T0 + (-alpha*s*std::sin(magb*q*s/p)/magb + alpha*p*(1 - std::cos(magb*q*s/p))/(std::pow(magb, 2)*q))*N0 + (-gamma*p*(q*s*std::cos(magb*q*s/p)/p - q*s/p)/(magb*q) + gamma*p*(-magb*q*s/p + std::sin(magb*q*s/p))/(std::pow(magb, 2)*q))*h;

        const AlgebraicVector3 dPdB = p*((-q*s*std::sin(magb*q*s/p)/p)*T0 + (-alpha*q*s*std::cos(magb*q*s/p)/p)*N0 + (gamma*q*s*std::sin(magb*q*s/p)/p)*h);

        AlgebraicVector6 dFglobal;
        dFglobal(0) = dMdB(0);
        dFglobal(1) = dMdB(1);
        dFglobal(2) = dMdB(2);
        dFglobal(3) = dPdB(0);
        dFglobal(4) = dPdB(1);
        dFglobal(5) = dPdB(2);

        //convert to curvilinear
        JacobianCartesianToCurvilinear cart2curv(propresult.first.globalParameters());
        AlgebraicVector5 iDF = cart2curv.jacobian()*dFglobal;
        
        //convert to tesla
        iDF *= 2.99792458e-3;
        
        dF.block<5,1>(5*i, bfieldidx) = Map<const Vector5d>(iDF.Array());
        
//         std::cout << "dF = " << dF.block<5,1>(5*i, bfieldidx) << std::endl;
        
//         const unsigned int bfieldglobalidx = std::lower_bound(detidparms.begin(), detidparms.end(), std::make_pair(2,hit->geographicalId())) - detidparms.begin();
//         assert(bfieldglobalidx<detidparms.size());
        const unsigned int bfieldglobalidx = detidparms[std::make_pair(2,hit->geographicalId())];
        globalidxv[nparsAlignment + bfieldidx] = bfieldglobalidx;
        bfieldidx++;
        
        //apply material effects (in reverse)
        TrajectoryStateOnSurface tmptsos(propresult.first.localParameters(),
                                          LocalTrajectoryError(),
                                          propresult.first.surface(),
                                          propresult.first.magneticField(),
                                          SurfaceSideDefinition::afterSurface);
        
        rPropagator.materialEffectsUpdator().updateStateInPlace(tmptsos, rpropdir);
        
        JacobianCurvilinearToLocal hprop(tmptsos.surface(), tmptsos.localParameters(), *tmptsos.magneticField());
        const AlgebraicMatrix55 &jachprop = hprop.jacobian();
        //efficient assignment from SMatrix using Eigen::Map
        Hprop.block<5,5>(5*i, 5*i) = Map<const Matrix<double, 5, 5, RowMajor> >(jachprop.Array());
        
        //full analytic energy loss jacobian (gross)
        //n.b this is the jacobian in LOCAL parameters (so E multiplies to the left of H s.t the total projection is E*Hprop*F)
        const double m2 = pow(rPropagator.materialEffectsUpdator().mass(), 2);  // use mass hypothesis from constructor
        constexpr double emass = 0.511e-3;
        constexpr double poti = 16.e-9 * 10.75;                 // = 16 eV * Z**0.9, for Si Z=14
        const double eplasma = 28.816e-9 * sqrt(2.33 * 0.498);  // 28.816 eV * sqrt(rho*(Z/A)) for Si
        const double qop = propresult.first.localParameters().qbp();
        const double dxdz = propresult.first.localParameters().dxdz();
        const double dydz = propresult.first.localParameters().dydz();
        const double xi = updtsos.surface().mediumProperties().xi();
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

        E(5*i,5*i) = res_0;
        E(5*i,5*i+1) = res_1;
        E(5*i,5*i+2) = res_2;
        E(5*i+1,5*i+1) = 1.;
        E(5*i+2,5*i+2) = 1.;
        E(5*i+3,5*i+3) = 1.;
        E(5*i+4,5*i+4) = 1.;
        
//         std::cout << "Eloss 0,0 = " << E(5*i,5*i) << std::endl;
        
        //derivative of the energy loss with respect to the energy loss parameter xi
        dE(5*i, elossidx) = res_3;
//         const unsigned int elossglobalidx = std::lower_bound(detidparms.begin(), detidparms.end(), std::make_pair(3,hit->geographicalId())) - detidparms.begin();
//         assert(elossglobalidx<detidparms.size());
        const unsigned int elossglobalidx = detidparms[std::make_pair(3,hit->geographicalId())];        
        globalidxv[nparsAlignment + nparsBfield + elossidx] = elossglobalidx;
        elossidx++;

        //momentum kink residual
        AlgebraicVector5 const& idx0 = updtsos.localParameters().vector() - tmptsos.localParameters().vector();
        dx0.segment<5>(5*i) = Map<const Vector5d>(idx0.Array());
        
        AlgebraicMatrix55 const Qmat = tmptsos.localError().matrix();
        Map<const Matrix<double, 5, 5, RowMajor> >iQ(Qmat.Array());
        //Q is 3x3 in the upper left block because there is no displacement on thin scattering layers
        //so invert the upper 3x3 block and insert small values by hand for the displacement uncertainties
        const double epsxy = 1e-5; //0.1um
        Qinv.block<3,3>(5*i,5*i) = iQ.topLeftCorner<3,3>().inverse();
        Qinv(5*i+3, 5*i+3) = 1./epsxy/epsxy;
        Qinv(5*i+4, 5*i+4) = 1./epsxy/epsxy;

//         std::cout << "Qinv" << std::endl;        
//         std::cout << Qinv.block<5,5>(5*i, 5*i) << std::endl; 

        
        //TODO compute G
        
      }

    }
    
    //now do the expensive calculations
    
    //expressions for gradients semi-automatically generated with sympy
    const GlobalParameterMatrix C = (2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H) + 2*H.transpose()*Vinv*H).ldlt().solve(GlobalParameterMatrix::Identity(5*nhits,5*nhits));
  
    //set gradients
    grad.head(nparsAlignment) = -2*A.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0) - 4*A.transpose()*Vinv*H*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*A.transpose()*Vinv*H*C.transpose()*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0);
    
    grad.segment(nparsAlignment,nparsBfield) = -2*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) - 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0);
    
    grad.tail(nparsEloss) = -2*dE.transpose()*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) - 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0);
    
    //fill hessian (diagonal blocks)
    hess.topLeftCorner(nparsAlignment, nparsAlignment) = 2*A.transpose()*Vinv*A - 4*A.transpose()*Vinv*H*C*H.transpose()*Vinv*A + 8*A.transpose()*Vinv*H*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 4*A.transpose()*Vinv*H*C.transpose()*H.transpose()*Vinv*A + 8*A.transpose()*Vinv*H*C.transpose()*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A;
    
    hess.block(nparsAlignment, nparsAlignment, nparsBfield, nparsBfield) = -4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF - 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 2*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*E*Hprop*dF;
    
    hess.bottomRightCorner(nparsEloss,nparsEloss) = -4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE - 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE + 2*dE.transpose()*Qinv*dE;
    
    //fill hessian off-diagonal blocks (lower triangular part)
    hess.block(nparsAlignment, 0, nparsBfield, nparsAlignment) = 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A + 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*A - 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A;
    
    hess.block(nparsAlignment+nparsBfield, 0, nparsEloss, nparsAlignment) = 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A + 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A;
    
    //careful this is easy to screw up because it is "accidentally" symmetric
    hess.block(nparsAlignment+nparsBfield, nparsAlignment, nparsEloss, nparsBfield) = -4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF - 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C.transpose()*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 2*dE.transpose()*Qinv*E*Hprop*dF;
    
    //fill upper triangular blocks
    hess.transpose().block(nparsAlignment, 0, nparsBfield, nparsAlignment) = hess.block(nparsAlignment, 0, nparsBfield, nparsAlignment);
    
    hess.transpose().block(nparsAlignment+nparsBfield, 0, nparsEloss, nparsAlignment) = hess.block(nparsAlignment+nparsBfield, 0, nparsEloss, nparsAlignment);
    
    hess.transpose().block(nparsAlignment+nparsBfield, nparsAlignment, nparsEloss, nparsBfield) = hess.block(nparsAlignment+nparsBfield, nparsAlignment, nparsEloss, nparsBfield);
    
//     std::cout << Qinv << std::endl;
    std::cout << "Recomputed innner covariance:" << std::endl;
    std::cout << 2.*C.bottomRightCorner<5,5>() << std::endl;
    
    std::cout << "KF innner covariance:" << std::endl;
    std::cout << tms[nhits-1].updatedState().curvilinearError().matrix() << std::endl;

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
  
//   std::map<std::pair<int,DetId>, std::tuple<int, int, int, int> > detidmap;
  std::set<std::tuple<int, int, int, int> > keys;
  
  for (const GeomDet* det : globalGeometry->dets()) {
//     printf("GeomDet pointer %p\n", det);
    if (!det) {
      continue;
    }
    if (!GeomDetEnumerators::isTracker(det->subDetector())) {
      continue;
    }
    int layer = 0;
    int subdet = det->subDetector();
    float eta = det->surface().position().eta();
    int ieta = std::floor(det->surface().position().eta()/0.2);

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
    
    for (int j=0; j<4; ++j) {
      if ( j==1 && !GeomDetEnumerators::isTrackerPixel(det->subDetector()) ) {
        continue;
      }
      auto const& key = std::make_tuple(j, subdet, layer, ieta);
      keys.insert(key);
    }
    

    
//     detidmap.emplace(std::make_pair(std::make_tuple(ieta, subdet, layer), det->geographicalId()));
    
    
//     if (GeomDetEnumerators::isTracker(det->subDetector())) {
//       //always have parameters for local x alignment, bfield, and e-loss
//       detidparms.emplace_back(0, det->geographicalId());
//       detidparms.emplace_back(2, det->geographicalId());
//       detidparms.emplace_back(3, det->geographicalId());
//       if (GeomDetEnumerators::isTrackerPixel(det->subDetector())) {
//         //local y alignment parameters only for pixels for now
//         detidparms.emplace_back(1, det->geographicalId());
//       }
//     }
    
  }
  
  nglobalparmsalignment = 0;
  nglobalparmsbfield = 0;
  nglobalparmseloss = 0;
  
  std::vector<std::tuple<int, int, int, int> > keysv;
  for (const auto& key : keys) {
    keysv.push_back(key);
    int parmtype = std::get<0>(key);
    if (parmtype < 2) {
      nglobalparmsalignment++;
    }
    else if (parmtype == 2) {
      nglobalparmsbfield++;
    }
    else if (parmtype == 3) {
      nglobalparmseloss++;
    }
  }
  std::sort(keysv.begin(), keysv.end());
  
  unsigned int badidx = 576;
  std::cout << "parameter " << badidx << ":" << std::endl;
  std::cout << std::get<0>(keysv[badidx]) << ", " << std::get<1>(keysv[badidx]) << ", " << std::get<2>(keysv[badidx]) << ", " << std::get<3>(keysv[badidx]) << std::endl;
  
  TTree *runtree = new TTree("runtree", "runtree");
  
  int iidx;
  int parmtype;
  int subdet;
  int layer;
  int ieta;

  runtree->Branch("iidx", &iidx);
  runtree->Branch("parmtype", &parmtype);
  runtree->Branch("subdet", &subdet);
  runtree->Branch("layer", &layer);
  runtree->Branch("ieta", &ieta);
  
//   for auto const& key : keysv
  for (unsigned int i=0; i<keysv.size(); ++i) {
    iidx = i;
    auto const& key = keysv[i];
    parmtype = std::get<0>(key);
    subdet = std::get<1>(key);
    layer = std::get<2>(key);
    ieta = std::get<3>(key);
    
    runtree->Fill();
  }
  

  for (const GeomDet* det : globalGeometry->dets()) {
//     printf("GeomDet pointer %p\n", det);
    if (!det) {
      continue;
    }
    if (!GeomDetEnumerators::isTracker(det->subDetector())) {
      continue;
    }
    int layer = 0;
    int subdet = det->subDetector();
    float eta = det->surface().position().eta();
    int ieta = std::floor(det->surface().position().eta()/0.2);

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
    
    for (int j=0; j<4; ++j) {
      if ( j==1 && !GeomDetEnumerators::isTrackerPixel(det->subDetector()) ) {
        continue;
      }
      auto const& key = std::make_tuple(j, subdet, layer, ieta);
      detidparms[std::make_pair(j, det->geographicalId())] = std::lower_bound(keysv.begin(),keysv.end(),key)-keysv.begin();
    }
    
//     detidmap.emplace(std::make_pair(std::make_tuple(ieta, subdet, layer), det->geographicalId()));
    
    
//     if (GeomDetEnumerators::isTracker(det->subDetector())) {
//       //always have parameters for local x alignment, bfield, and e-loss
//       detidparms.emplace_back(0, det->geographicalId());
//       detidparms.emplace_back(2, det->geographicalId());
//       detidparms.emplace_back(3, det->geographicalId());
//       if (GeomDetEnumerators::isTrackerPixel(det->subDetector())) {
//         //local y alignment parameters only for pixels for now
//         detidparms.emplace_back(1, det->geographicalId());
//       }
//     }
    
  }
  
//   for (const GeomDet* det : globalGeometry->dets()) {
// //     printf("GeomDet pointer %p\n", det);
//     if (!det) {
//       continue;
//     }
//     if (!GeomDetEnumerators::isTracker(det->subDetector())) {
//       continue;
//     }
//     
//     detidparms[det->geographicalId()] = keys.find[detidmap[det->geographicalId()]]-keys.begin();
//   }
//   
  
  
//   std::sort(detidparms.begin(), detidparms.end());
  nglobalparms = keys.size();
  std::cout << "nglobalparms = " << nglobalparms << std::endl;
    
  
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

//define this as a plug-in
DEFINE_FWK_MODULE(ResidualGlobalCorrectionMaker);
