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
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"

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

//
// class declaration
//

class HitAnalyzer : public edm::EDAnalyzer
{
public:
  explicit HitAnalyzer(const edm::ParameterSet &);
  ~HitAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  //  edm::EDGetTokenT<reco::TrackCollection>      inputTracks_;
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;

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
  std::vector<std::vector<double>> trackF;
  std::vector<std::vector<double>> trackC;

  std::vector<std::vector<double>> updState;
  std::vector<std::vector<double>> backPropState;
  std::vector<std::vector<double>> forwardPropState;

  std::vector<std::vector<double>> curvpars;

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
HitAnalyzer::HitAnalyzer(const edm::ParameterSet &iConfig)

{
  //now do what ever initialization is needed
  //  inputTracks_ = consumes<reco::TrackCollection>(edm::InputTag("TrackRefitter"));
  inputTraj_ = consumes<std::vector<Trajectory>>(edm::InputTag("TrackRefitter"));
  GenParticlesToken_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));

  n = 0;

  fout = new TFile("trackTree.root", "RECREATE");
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
  tree->Branch("trackF", &trackF);
  tree->Branch("trackC", &trackC);
  tree->Branch("updState", &updState);
  tree->Branch("backPropState", &backPropState);
  tree->Branch("forwardPropState", &forwardPropState);
  tree->Branch("curvpars", &curvpars);
}

HitAnalyzer::~HitAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HitAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
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

  ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);
  auto field = magfield.product();

  for (unsigned int j = 0; j < trajH->size(); ++j)
  {

    //     const reco::Track& track = (*trackH)[i];
    //     if (track.lost()>0)
    //       continue;

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

    //material stuff
    trackQ.clear();
    trackH.clear();
    trackF.clear();
    trackC.clear();

    const float mass = 0.1395703;
    const float maxDPhi = 1.6;
    PropagatorWithMaterial Propagator((*trajH)[j].direction(), mass, field, maxDPhi, true, -1., false);

    for (unsigned int i = 0; i < tms.size(); ++i)
    {
      TrajectoryStateOnSurface updatedState = tms[i].updatedState();

      if (!updatedState.isValid())
        continue;

      if (!tms[i].recHit()->isValid()){
        ninvalidHits++;
        continue;
      }
      stereo.push_back(0);
      glued.push_back(0);
      pt.push_back(updatedState.globalMomentum().perp());

      const GeomDet *detectorG = globalGeometry->idToDet(tms[i].recHit()->geographicalId());
      LocalPoint local(tms[i].recHit()->localPosition().x(), tms[i].recHit()->localPosition().y(), tms[i].recHit()->localPosition().z());

      if (detectorG->subDetector() == GeomDetEnumerators::PixelBarrel)
      {
        PXBDetId detid(tms[i].recHit()->rawId());
        layer.push_back(detid.layer());
      }
      else if (detectorG->subDetector() == GeomDetEnumerators::PixelEndcap)
      {
        PXFDetId detid(tms[i].recHit()->rawId());
        layer.push_back(-1 * (detid.side() == 1) * detid.disk() + (detid.side() == 2) * detid.disk());
      }
      else if (detectorG->subDetector() == GeomDetEnumerators::TIB)
      {
        TIBDetId detid(tms[i].recHit()->rawId());
        layer.push_back(detid.layer());
        if (detid.stereo() != 0)
          stereo.push_back(1);
        if (detid.glued() != 0)
          glued.push_back(1);
        local = LocalPoint(tms[i].recHit()->localPosition().x(), updatedState.localPosition().y(), tms[i].recHit()->localPosition().z());
      }
      else if (detectorG->subDetector() == GeomDetEnumerators::TOB)
      {
        TOBDetId detid(tms[i].recHit()->rawId());
        layer.push_back(detid.layer());
        if (detid.stereo() != 0)
          stereo.push_back(1);
        if (detid.glued() != 0)
          glued.push_back(1);

        local = LocalPoint(tms[i].recHit()->localPosition().x(), updatedState.localPosition().y(), tms[i].recHit()->localPosition().z());
      }
      else if (detectorG->subDetector() == GeomDetEnumerators::TID)
      {
        TIDDetId detid(tms[i].recHit()->rawId());
        layer.push_back(-1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel());
        if (detid.stereo() != 0)
          stereo.push_back(1);
        if (detid.glued() != 0)
          glued.push_back(1);

        const StripTopology *theTopology = dynamic_cast<const StripTopology *>(&(tms[i].recHit()->detUnit()->topology()));
        MeasurementPoint point = theTopology->measurementPosition(tms[i].recHit()->localPosition());
        MeasurementPoint pointT = theTopology->measurementPosition(updatedState.localPosition());
        MeasurementPoint pointC(point.x(), pointT.y());
        local = theTopology->localPosition(pointC);
      }
      else if (detectorG->subDetector() == GeomDetEnumerators::TEC)
      {
        TECDetId detid(tms[i].recHit()->rawId());
        layer.push_back(-1 * (detid.side() == 1) * detid.wheel() + (detid.side() == 2) * detid.wheel());
        if (detid.stereo() != 0)
          stereo.push_back(1);
        if (detid.glued() != 0)
          glued.push_back(1);

        const StripTopology *theTopology = dynamic_cast<const StripTopology *>(&(tms[i].recHit()->detUnit()->topology()));
        MeasurementPoint point = theTopology->measurementPosition(tms[i].recHit()->localPosition());
        MeasurementPoint pointT = theTopology->measurementPosition(updatedState.localPosition());
        MeasurementPoint pointC(point.x(), pointT.y());
        local = theTopology->localPosition(pointC);
      }

      // material info

      // Get surface
      const Surface &surface = updatedState.surface();
      // Now get information on medium
      const MediumProperties &mp = surface.mediumProperties(); // parsed from xml tables

      // Momentum vector
      LocalVector d = updatedState.localMomentum();
      float p2 = d.mag2();
      d *= 1.f / sqrt(p2);
      float xf = 1.f / std::abs(d.z()); // increase of path due to angle of incidence
      // calculate general physics things
      constexpr float amscon = 1.8496e-4; // (13.6MeV)**
      const float m2 = mass * mass;       // use mass hypothesis from constructor
      float e2 = p2 + m2;
      float beta2 = p2 / e2;
      // calculate the multiple scattering angle
      float radLen = mp.radLen() * xf; // effective rad. length
      float sigt2 = 0.;                // sigma(alpha)**2

      // Calculated rms scattering angle squared.
      float fact = 1.f + 0.038f * unsafe_logf<2>(radLen);
      fact *= fact;
      float a = fact / (beta2 * p2);
      sigt2 = amscon * radLen * a;

      float isl2 = 1.f / d.perp2();
      float cl2 = (d.z() * d.z());
      float cf2 = (d.x() * d.x()) * isl2;
      float sf2 = (d.y() * d.y()) * isl2;
      // Create update (transformation of independant variations
      //   on angle in orthogonal planes to local parameters.
      float den = 1.f / (cl2 * cl2);

      float msxx = (den * sigt2) * (sf2 * cl2 + cf2);
      float msxy = (den * sigt2) * (d.x() * d.y());
      float msyy = (den * sigt2) * (cf2 * cl2 + sf2);

      //energy loss
      constexpr float emass = 0.511e-3;
      constexpr float poti = 16.e-9 * 10.75;                // = 16 eV * Z**0.9, for Si Z=14
      const float eplasma = 28.816e-9 * sqrt(2.33 * 0.498); // 28.816 eV * sqrt(rho*(Z/A)) for Si
      const float delta0 = 2 * log(eplasma / poti) - 1.;

      // calculate general physics things
      float im2 = float(1.) / m2;
      float e = std::sqrt(e2);
      float eta2 = p2 * im2;
      float ratio2 = (emass * emass) * im2;
      float emax = float(2.) * emass * eta2 / (float(1.) + float(2.) * emass * e * im2 + ratio2);

      float xi = mp.xi() * xf;
      xi /= beta2;

      float dEdx = xi * (unsafe_logf<2>(float(2.) * emass * emax / (poti * poti)) - float(2.) * (beta2)-delta0);

      float dEdx2 = xi * emax * (float(1.) - float(0.5) * beta2);
      float dPoverP = dEdx / std::sqrt(beta2) / sqrt(p2);
      float sigp2 = dEdx2 / (beta2 * p2 * p2);

      Matrix3d Qmat;

      Qmat << sigp2, 0., 0.,
          0., msxx, msxy,
          0., msxy, msyy;

      std::vector<double> Q;
      Q.resize(9);
      Eigen::Map<Matrix3d>(Q.data(), 3, 3) = Qmat;

      trackQ.push_back(Q);

      // get the transformation matrix from local coordinates to helix parameters
      auto &x = updatedState.localParameters();

      JacobianLocalToCurvilinear local2curv(surface, x, *field);
      const AlgebraicMatrix55 &jac = local2curv.jacobian();

      Matrix5d jac_copy;
      jac_copy << jac[0][0], jac[0][1], jac[0][2], jac[0][3], jac[0][4],
          jac[1][0], jac[1][1], jac[1][2], jac[1][3], jac[1][4],
          jac[2][0], jac[2][1], jac[2][2], jac[2][3], jac[2][4],
          jac[3][0], jac[3][1], jac[3][2], jac[3][3], jac[3][4],
          jac[4][0], jac[4][1], jac[4][2], jac[4][3], jac[4][4];

      std::vector<double> H;
      H.resize(25);
      Eigen::Map<Matrix5d>(H.data(), 5, 5) = jac_copy;

      trackH.push_back(H);

      const AlgebraicSymMatrix55 &covariance = updatedState.curvilinearError().matrix();
      Matrix5d jac_copy3;
      jac_copy3 << covariance[0][0], covariance[0][1], covariance[0][2], covariance[0][3], covariance[0][4],
          covariance[1][0], covariance[1][1], covariance[1][2], covariance[1][3], covariance[1][4],
          covariance[2][0], covariance[2][1], covariance[2][2], covariance[2][3], covariance[2][4],
          covariance[3][0], covariance[3][1], covariance[3][2], covariance[3][3], covariance[3][4],
          covariance[4][0], covariance[4][1], covariance[4][2], covariance[4][3], covariance[4][4];

      std::vector<double> C;
      C.resize(25);
      Eigen::Map<Matrix5d>(C.data(), 5, 5) = jac_copy3;

      trackC.push_back(C);

      LocalTrajectoryParameters updPars = updatedState.localParameters();
      //CurvilinearTrajectoryParameters curvPrs(updPars.position(), updPars.momentum(), trackCharge);
      AlgebraicVector5 curvPrsv = updPars.vector();

      Vector5d curvPrsv_copy;
      curvPrsv_copy << curvPrsv[0], curvPrsv[1], curvPrsv[2], curvPrsv[3], curvPrsv[4];
      std::vector<double> curvPrsv_vec;
      curvPrsv_vec.resize(5);
      Eigen::Map<Vector5d>(curvPrsv_vec.data(), 5, 1) = curvPrsv_copy;
      updState.push_back(curvPrsv_vec);

      if (i == 0)
      {
        Vector5d backcurvPrsv_copy = Vector5d::Zero(5);
        std::vector<double> backcurvPrsv_vec;
        backcurvPrsv_vec.resize(5);
        Eigen::Map<Vector5d>(backcurvPrsv_vec.data(), 5, 1) = backcurvPrsv_copy;
        backPropState.push_back(backcurvPrsv_vec);

        Matrix5d jac_copy2 = Matrix5d::Zero(5,5);
        std::vector<double> F;
        F.resize(25);
        Eigen::Map<Matrix5d>(F.data(), 5, 5) = jac_copy2;

        trackF.push_back(F);
      }
      else
      {
        const auto &propresult = Propagator.propagateWithPath(tms[i-1].updatedState(), tms[i].updatedState().surface());
        LocalTrajectoryParameters backPars = propresult.first.localParameters();
        AlgebraicVector5 backcurvPrsv = backPars.vector();

        Vector5d backcurvPrsv_copy;
        backcurvPrsv_copy << backcurvPrsv[0], backcurvPrsv[1], backcurvPrsv[2], backcurvPrsv[3], backcurvPrsv[4];
        std::vector<double> backcurvPrsv_vec;
        backcurvPrsv_vec.resize(5);
        Eigen::Map<Vector5d>(backcurvPrsv_vec.data(), 5, 1) = backcurvPrsv_copy;
        backPropState.push_back(backcurvPrsv_vec);

        AnalyticalCurvilinearJacobian curvjac;
        GlobalTrajectoryParameters tpg = tms[i - 1].updatedState().globalParameters();
        GlobalTrajectoryParameters tpg2 = tms[i].updatedState().globalParameters();
        GlobalVector h = tpg.magneticFieldInInverseGeV(tpg.position());
        curvjac.computeFullJacobian(tpg, tpg2.position(), tpg2.momentum(), h, propresult.second);
        const AlgebraicMatrix55 &jac2 = curvjac.jacobian();

        Matrix5d jac_copy2;
        jac_copy2 << jac2[0][0], jac2[0][1], jac2[0][2], jac2[0][3], jac2[0][4],
            jac2[1][0], jac2[1][1], jac2[1][2], jac2[1][3], jac2[1][4],
            jac2[2][0], jac2[2][1], jac2[2][2], jac2[2][3], jac2[2][4],
            jac2[3][0], jac2[3][1], jac2[3][2], jac2[3][3], jac2[3][4],
            jac2[4][0], jac2[4][1], jac2[4][2], jac2[4][3], jac2[4][4];

        Matrix5d eloss = Matrix5d::Identity(5, 5);
        if (trackCharge > 0)
          eloss(0, 0) += dPoverP;
        else
          eloss(0, 0) -= dPoverP;

        jac_copy2 = eloss * jac_copy2;

        std::vector<double> F;
        F.resize(25);
        Eigen::Map<Matrix5d>(F.data(), 5, 5) = jac_copy2;

        trackF.push_back(F);
      }

      detector.push_back(detectorG->subDetector());
      GlobalPoint corrected = detectorG->toGlobal(local);
      z.push_back(corrected.z());
      eta.push_back(corrected.eta());
      phi.push_back(corrected.phi());
      r.push_back(corrected.perp());

      float xhyb = r[n] * TMath::Cos(phi[n]);
      float yhyb = r[n] * TMath::Sin(phi[n]);

      LocalVector perp(0., 0., 1.);
      GlobalVector globperp = detectorG->toGlobal(perp);
      GlobalPoint globpoint(xhyb, yhyb, z[n]);

      CurvilinearTrajectoryParameters cp(globpoint, globperp, trackCharge);

      AlgebraicVector5 cpvec =cp.vector();

      Vector5d vec_curv;
      vec_curv << cpvec[0], cpvec[1], cpvec[2], cpvec[3], cpvec[4];
      std::vector<double> curv;
      
      curv.resize(5);
      Eigen::Map<Vector5d>(curv.data(), 5, 1) = vec_curv;

      curvpars.push_back(curv);

      xUnc.push_back(tms[i].recHit()->globalPosition().x());
      yUnc.push_back(tms[i].recHit()->globalPosition().y());
      zUnc.push_back(tms[i].recHit()->globalPosition().z());
      etaUnc.push_back(tms[i].recHit()->globalPosition().eta());
      phiUnc.push_back(tms[i].recHit()->globalPosition().phi());
      rUnc.push_back(tms[i].recHit()->globalPosition().perp());

      localx.push_back(tms[i].recHit()->localPosition().x());
      localy.push_back(tms[i].recHit()->localPosition().y());
      localz.push_back(tms[i].recHit()->localPosition().z());

      localx_state.push_back(updatedState.localPosition().x());
      localy_state.push_back(updatedState.localPosition().y());

      localxErr.push_back(tms[i].recHit()->localPositionError().xx());
      localyErr.push_back(tms[i].recHit()->localPositionError().yy());
      localxyErr.push_back(tms[i].recHit()->localPositionError().xy());

      // globalrErr.push_back(tms[i].recHit()->errorGlobalR();
      // globalzErr.push_back(tms[i].recHit()->errorGlobalZ();
      // globalrphiErr.push_back(tms[i].recHit()->errorGlobalRPhi();
      GlobalError globalError = ErrorFrameTransformer::transform(tms[i].recHit()->localPositionError(), surface);
      globalrphiErr.push_back(r[n] * std::sqrt(float(globalError.phierr(corrected))));
      globalrErr.push_back(std::sqrt(float(globalError.rerr(corrected))));
      globalzErr.push_back(std::sqrt(float(globalError.czz())));

      n = n + 1;
    }

    tree->Fill();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HitAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void HitAnalyzer::endJob()
{
  fout->cd();
  fout->Write();
  fout->Close();
}

// ------------ method called when starting to processes a run  ------------
/*
void 
HitAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

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
void HitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HitAnalyzer);
