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

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<Trajectory>> inputTraj_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticlesToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> inputTrack_;
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
  inputBs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));


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
  tree->Branch("jacrefv",jacrefv.data(),"jacrefv[nParms]/F", basketSize);
  
  tree->Branch("nSym", &nSym, basketSize);
  
  tree->Branch("hesspackedv", hesspackedv.data(), "hesspackedv[nSym]/F", basketSize);

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
  
  Handle<std::vector<Trajectory> > trajH;
  iEvent.getByToken(inputTraj_, trajH);
  
  Handle<reco::BeamSpot> bsH;
  iEvent.getByToken(inputBs_, bsH);
  
  const reco::BeamSpot& bs = *bsH;
  
  const float mass = 0.105;
  const float maxDPhi = 1.6;
  PropagatorWithMaterial rPropagator(oppositeToMomentum, mass, field, maxDPhi, true, -1., false);
  PropagatorWithMaterial fPropagator(alongMomentum, mass, field, maxDPhi, true, -1., false);
  
  for (unsigned int j=0; j<trajH->size(); ++j) {
    const Trajectory& traj = (*trajH)[j];
    
    const edm::Ref<std::vector<Trajectory> > trajref(trajH, j);
    const reco::Track& track = *(*trackH)[trajref];

    if (traj.isLooper()) {
      continue;
    }
    trackPt = track.pt();
    trackEta = track.eta();
    trackPhi = track.phi();
    trackCharge = track.charge();
    trackPtErr = track.ptError();
    
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
    ELossJacobianMatrix dE = ELossJacobianMatrix::Zero(5*nhits, nparsEloss);
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
    
    
    globalidxv.clear();
    globalidxv.resize(npars, 0);
    
    nParms = npars;
    tree->SetBranchAddress("globalidxv", globalidxv.data());
    
    bool valid = true;
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
          //fill y residual
          dy0(5*i+4) = hit->localPosition().y() - updtsos.localPosition().y();
          
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
      const unsigned int xglobalidx = detidparms.at(std::make_pair(0,hit->geographicalId()));
      globalidxv[alignmentidx] = xglobalidx;
      alignmentidx++;
      if (ispixel) {
        A(5*i+4, alignmentidx) = 1.;
        const unsigned int yglobalidx = detidparms.at(std::make_pair(1,hit->geographicalId()));
        globalidxv[alignmentidx] = yglobalidx;
        alignmentidx++;
      }

      
      if (i >0) {        
        //compute transport jacobian propagating outside in to current layer
        TrajectoryStateOnSurface const& toproptsos = tms[i-1].updatedState();
        
        //n.b. the returned pathlength is negative when propagating oppositeToMomentum, and this ensures the correct
        //results for the transport Jacobians
        auto const& propresult = rPropagator.geometricalPropagator().propagateWithPath(toproptsos, updtsos.surface());
        if (!propresult.first.isValid()) {
          valid = false;
          break;
        }
        AnalyticalCurvilinearJacobian curvjac(toproptsos.globalParameters(), propresult.first.globalParameters().position(), propresult.first.globalParameters().momentum(), propresult.second);
        const AlgebraicMatrix55 &jacF = curvjac.jacobian();
        
        F.block<5,5>(5*i, 5*(i-1)) = Map<const Matrix<double, 5, 5, RowMajor> >(jacF.Array());
        
        //analytic jacobian wrt magnitude of magnetic field
        //TODO should we parameterize with respect to z-component instead?
        //extending derivation from CMS NOTE 2006/001
        const Vector3d b(toproptsos.globalParameters().magneticFieldInInverseGeV().x(),
                           toproptsos.globalParameters().magneticFieldInInverseGeV().y(),
                           toproptsos.globalParameters().magneticFieldInInverseGeV().z());
        double magb = b.norm();
        const Vector3d h = b.normalized();

        const Vector3d p0(toproptsos.globalParameters().momentum().x(),
                            toproptsos.globalParameters().momentum().y(),
                            toproptsos.globalParameters().momentum().z());
        const Vector3d p1(propresult.first.globalParameters().momentum().x(),
                            propresult.first.globalParameters().momentum().y(),
                            propresult.first.globalParameters().momentum().z());
        const Vector3d T0 = p0.normalized();
        double p = p1.norm();
        const Vector3d T = p1.normalized();
        double q = toproptsos.charge();
        
        const Vector3d N0 = h.cross(T0).normalized();
        const double alpha = h.cross(T).norm();
        const double gamma = h.transpose()*T;
        const double s = propresult.second;

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
        JacobianCartesianToCurvilinear cart2curv(propresult.first.globalParameters());
        const AlgebraicMatrix56& cart2curvjacs = cart2curv.jacobian();
        const Map<const Matrix<double, 5, 6, RowMajor> > cart2curvjac(cart2curvjacs.Array());
        
        //compute final jacobian (and convert to Tesla)
        dF.block<5,1>(5*i, bfieldidx) = 2.99792458e-3*cart2curvjac*dFglobal;
        
//         std::cout << "dF = " << dF.block<5,1>(5*i, bfieldidx) << std::endl;
        
        const unsigned int bfieldglobalidx = detidparms.at(std::make_pair(2,hit->geographicalId()));
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
        const unsigned int elossglobalidx = detidparms.at(std::make_pair(3,hit->geographicalId()));
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

                
      }

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
      continue;
    }
    AnalyticalCurvilinearJacobian curvjac(propresult.first.globalParameters(), refFts.position(), refFts.momentum(), -propresult.second);
    const AlgebraicMatrix55& jacF = curvjac.jacobian();
    Map<const Matrix<double, 5, 5, RowMajor> > Fref(jacF.Array());

    //now do the expensive calculations and fill outputs
    
    gradv.clear();
    jacrefv.clear();

    gradv.resize(npars,0.);
    jacrefv.resize(5*npars, 0.);
    
    nJacRef = 5*npars;
    tree->SetBranchAddress("gradv", gradv.data());
    tree->SetBranchAddress("jacrefv", jacrefv.data());
    
    //eigen representation of the underlying vector storage
    Map<VectorXf> grad(gradv.data(), npars);
    Map<Matrix<float, 5, Dynamic, RowMajor> > jacref(jacrefv.data(), 5, npars);
    
    //hessian is not stored directly so it's a standard Eigen Matrix
    MatrixXf hess = MatrixXf::Zero(npars, npars);
    
    //expressions for gradients semi-automatically generated with sympy

    //compute robust cholesky decomposition of inverse covariance matrix
    auto const& Cinvldlt = (2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H) + 2*H.transpose()*Vinv*H).ldlt();
    
    //compute covariance matrix explicitly to simplify below expressions
    //TODO avoid this by replacing matrix multiplications with C with the appropriate ldlt solve operations in the gradients below
    const GlobalParameterMatrix C = Cinvldlt.solve(GlobalParameterMatrix::Identity(5*nhits,5*nhits));
    
    //compute shift in parameters at reference point from matrix model given hit and kink residuals
    //(but for nominal values of the model correction
    Vector5d dxRef = Fref*Cinvldlt.solve(-2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 + 2*H.transpose()*Vinv*dy0).tail<5>();
    
    //fill output with corrected state and covariance at reference point
    refParms.fill(0.);
    refCov.fill(0.);
    const AlgebraicVector5& refVec = track.parameters();
    Map<Vector5f>(refParms.data()) = (Map<const Vector5d>(refVec.Array()) + dxRef).cast<float>();
    Map<Matrix<float, 5, 5, RowMajor> >(refCov.data()).triangularView<Upper>() = (2.*Fref*C.bottomRightCorner<5,5>()*Fref.transpose()).cast<float>().triangularView<Upper>();
    
    //set jacobian for reference point parameters
    jacref.leftCols(nparsAlignment) = (-2*Fref*Cinvldlt.solve(H.transpose()*Vinv*A).bottomRows<5>()).cast<float>();

    jacref.block<5, Dynamic>(0, nparsAlignment, 5, nparsBfield) = (2*Fref*Cinvldlt.solve((-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF).bottomRows<5>()).cast<float>();
    
    jacref.rightCols(nparsEloss) = (2*Fref*Cinvldlt.solve((-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE).bottomRows<5>()).cast<float>();
    
//     std::cout << "jacref:" << std::endl;
//     std::cout << jacref << std::endl;
    
    //set gradients
    grad.head(nparsAlignment) = (-2*A.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0) - 4*A.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*A.transpose()*Vinv*H*C*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0)).cast<float>();

    grad.segment(nparsAlignment,nparsBfield) = (-2*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) - 4*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0)).cast<float>();
    
    grad.tail(nparsEloss) = (-2*dE.transpose()*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) + 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-(-E*Hprop*F + H)*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dx0) - 4*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*(H*C*(2*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dx0 - 2*H.transpose()*Vinv*dy0) + dy0)).cast<float>();
    
    //fill hessian (diagonal blocks, upper triangular part only)
    hess.topLeftCorner(nparsAlignment, nparsAlignment).triangularView<Upper>() = (2*A.transpose()*Vinv*A + 8*A.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 8*A.transpose()*Vinv*H*C*H.transpose()*Vinv*A + 8*A.transpose()*Vinv*H*C*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A).cast<float>().triangularView<Upper>();
    
    hess.block(nparsAlignment, nparsAlignment, nparsBfield, nparsBfield).triangularView<Upper>() = (8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF - 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 2*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*E*Hprop*dF).cast<float>().triangularView<Upper>();
    
    hess.bottomRightCorner(nparsEloss,nparsEloss).triangularView<Upper>() = (8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE - 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*dE + 2*dE.transpose()*Qinv*dE).cast<float>().triangularView<Upper>();
    
    //fill hessian off-diagonal blocks (upper triangular part)
    hess.transpose().block(nparsAlignment, 0, nparsBfield, nparsAlignment) = (-8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A + 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 8*dF.transpose()*Hprop.transpose()*E.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A).cast<float>();
    
    hess.transpose().block(nparsAlignment+nparsBfield, 0, nparsEloss, nparsAlignment) = (-8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*A - 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*H*C*H.transpose()*Vinv*A).cast<float>();
    
    //careful this is easy to screw up because it is "accidentally" symmetric
    hess.transpose().block(nparsAlignment+nparsBfield, nparsAlignment, nparsEloss, nparsBfield) = (8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF - 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 8*dE.transpose()*Qinv*(-E*Hprop*F + H)*C*H.transpose()*Vinv*H*C*(-F.transpose()*Hprop.transpose()*E.transpose() + H.transpose())*Qinv*E*Hprop*dF + 2*dE.transpose()*Qinv*E*Hprop*dF).cast<float>();
    
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
      hesspacked.segment(packedidx, segmentsize) = hess.block<1, Dynamic>(ipar, ipar, 1, segmentsize);
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

//define this as a plug-in
DEFINE_FWK_MODULE(ResidualGlobalCorrectionMaker);
