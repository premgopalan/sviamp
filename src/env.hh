#ifndef ENV_HH
#define ENV_HH

#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <error.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <map>
#include <list>
#include <vector>
#include "matrix.hh"
#include "log.hh"

typedef uint8_t yval_t;

typedef D2Array<yval_t> AdjMatrix;
typedef D2Array<double> Matrix;
typedef D3Array<double> D3;

typedef D1Array<KV> KVArray;
typedef D2Array<KV> MatrixKV;
typedef map<uint32_t, KV *> KVMap;
typedef list<KV *> KVList;

typedef std::map<uint32_t, uint32_t> IDMap;
typedef std::map<uint32_t, double> DoubleMap;
typedef std::map<uint32_t, uint32_t> FreqMap;
typedef std::map<string, uint32_t> FreqStrMap;
typedef std::map<string, uint32_t> StrMap;
typedef std::map<uint32_t, string> StrMapInv;
//typedef std::map<uint32_t, vector<uint32_t> > SparseMatrix;
typedef D1Array<std::vector<uint32_t> *> SparseMatrix;
typedef std::vector<Edge> EdgeList;
typedef std::map<uint32_t, bool> NodeMap;
typedef std::map<uint32_t, bool> BoolMap;
typedef std::map<uint32_t, uint32_t> NodeValMap;
typedef std::map<uint32_t, vector<uint32_t> > MapVec;
typedef std::map<uint32_t, vector<KV> > MapVecKV;
typedef MapVec SparseMatrix2;
typedef std::map<Edge, bool> SampleMap;
typedef std::map<Edge, int> CountMap;
typedef std::map<Edge, double> ValueMap;
typedef std::map<uint32_t, string> StrMapInv;
typedef D1Array<IDMap> KMap;

class Env {
public:
  Env(uint32_t N, uint32_t K, bool massive, 
      double hol_ratio, double rand_seed,      
      bool sbm, bool batch, bool strat, bool nodelay, 
      bool rpair, bool rnode, bool load, bool adamic,
      uint32_t scale,
      bool dis, bool force_overwrite_dir, string dfname, 
      bool ppc, bool run_gap, bool gen, string label,
      uint32_t nthrs, uint32_t itype, string etype,
      bool nmi, string ground_truth_fname, uint32_t rfreq, 
      bool accuracy, double sth, double inf, bool nu, 
      bool bmark, bool randzeros, bool preprocess, 
      bool strid, string groups_fname, 
      bool glm, bool glm2, bool pcp, 
      bool postprocess,
      bool accopt, bool lcaccopt, double ngscale, double link_thresh,
      uint32_t lt_min_deg, bool lc, bool nolambda, 
      uint32_t nmemberships, bool ammopt, bool onesonly,
      bool init_comm, string init_comm_fname,
      bool node_scaling_on, bool lpmode,
      bool gtrim, bool fastinit, uint32_t max_iterations,
      bool globalmu, bool adagrad);
  ~Env() { fclose(_plogf); }

  static string prefix;
  static Logger::Level level;

  uint32_t n;
  uint32_t k;
  uint32_t t;
  uint32_t s;
  uint32_t m;
  uint32_t sets_mini_batch;

  bool informative_sampling;
  bool single;
  bool batch_mode;
  
  int illegal_likelihood;
  int max_draw_edges;
  double meanchangethresh;
  double alpha;
  double sbm_alpha;
  double infer_alpha;
  bool model_load;
  bool adamic_adar;
  uint32_t subsample_scale;

  double eta0;
  double eta1;
  double heldout_ratio;
  double seed;
  double precision_ratio;

  double eta0_dense;
  double eta1_dense;
  double eta0_regular;
  double eta1_regular;
  double eta0_uniform;
  double eta1_uniform;
  double eta0_sparse;
  double eta1_sparse;
  double eta0_gen;
  double eta1_gen;
  
  int reportfreq;
  double epsilon;
  double logepsilon;

  double tau0;
  double nodetau0;
  double nodekappa;
  double kappa;
  uint32_t online_iterations;
  uint32_t conv_nupdates;
  double conv_thresh1;
  double conv_thresh2;

  bool stratified;
  bool delaylearn;
  bool undirected;
  bool randompair;
  bool randomnode;
  bool bfs;
  uint32_t bfs_nodes;
  bool citation;
  bool accuracy;
  double stopthresh;
  double infthresh;
  bool nonuniform;
  bool benchmark;
  bool randzeros;
  bool preprocess;
  bool strid;
  string groups_file;
  bool glm;
  bool glm2;
  bool pcp;
  bool postprocess;
  bool acc;
  bool lcacc;
  double ngscale;
  double link_thresh;
  uint32_t lt_min_deg;
  bool lowconf;
  bool nolambda;
  uint32_t nmemberships;
  bool amm;
  bool onesonly;
  bool terminate;
  
  string datdir;

  // debug
  bool deterministic;
  
  // gen
  bool disjoint;
  bool gen;
  
  // ppc
  uint32_t ppc_ndraws;

  // gamma poisson
  double default_shape;
  double default_rate;

  uint16_t nthreads;
  string label;
  string eta_type;

  bool nmi;
  string ground_truth_fname;

  bool use_validation_stop;
  bool use_training_stop;
  bool use_init_communities;
  string init_communities_fname;
  bool node_scaling_on;
  bool lpmode;
  bool gammatrim;
  bool log_training_likelihood;
  uint16_t max_iterations;

  bool globalmu;
  bool adagrad;

  template<class T> static void plog(string s, const T &v);
  static string file_str(string fname);

private:
  static FILE *_plogf;
};


template<class T> inline void
Env::plog(string s, const T &v)
{
  fprintf(_plogf, "%s: %s\n", s.c_str(), v.s().c_str());
  fflush(_plogf);  
}

template<> inline void
Env::plog(string s, const double &v)
{
  fprintf(_plogf, "%s: %.9f\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const bool &v)
{
  fprintf(_plogf, "%s: %s\n", s.c_str(), v ? "True": "False");
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const int &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const unsigned &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const short unsigned int &v)
{
  fprintf(_plogf, "%s: %d\n", s.c_str(), v);
  fflush(_plogf);
}

template<> inline void
Env::plog(string s, const uint64_t &v)
{
  fprintf(_plogf, "%s: %lu\n", s.c_str(), v);
  fflush(_plogf);
}

inline string
Env::file_str(string fname)
{
  string s = prefix + fname;
  return s;
}

inline
Env::Env(uint32_t N, uint32_t K, bool massive, 
	 double hol_ratio, double rand_seed,
	 bool sbm, bool batch, bool strat, bool nodelay, 
	 bool rpair, bool rnode, bool load, bool adamic,
	 uint32_t scale,
	 bool dis, bool force_overwrite_dir, string dfname, 
	 bool ppc, bool gap, bool gen1, string lbl,
	 uint32_t nthrs, uint32_t itype, string etype,
	 bool nmival, string gfname, uint32_t rfreq, 
	 bool accuracy, double sth, double inf, bool nu,
	 bool bmark, bool rzeros, bool pre, 
	 bool sid, string groups_fname, 
	 bool glmopt, bool glmopt2, bool pcpopt, 
	 bool pp,
	 bool accopt,
	 bool lcaccopt, double ng, double lt, 
	 uint32_t lt_min_deg_opt, bool lc, bool nol,
	 uint32_t nmem, bool ammopt, bool oo, bool init_comm,
	 string init_comm_fname, bool nscaling, bool lpm,
	 bool gtrim, bool fastinit, uint32_t max_itr,
	 bool gmu, bool agrad)
  : n(N),
    k(K),
    t(2),
    s(n/10),
    m(1),
    sets_mini_batch((double)n / 100),

    informative_sampling(massive),
    single(sbm),
    batch_mode(batch),
   
    illegal_likelihood(-1), 
    max_draw_edges(4096),
    meanchangethresh(0.00001),
    alpha((double)1.0/k),
    //alpha(1.0),
    //alpha((double)0.01),
    //alpha((double)1.0/1000),
    sbm_alpha(0.5),
    
    infer_alpha(alpha),
    model_load(load),
    adamic_adar(adamic),
    subsample_scale(scale),

    eta0(0), 
    eta1(0),
    heldout_ratio(hol_ratio),
    seed(rand_seed),
    precision_ratio(0.001),
    eta0_dense(4700.59),
    eta1_dense(0.77),
    eta0_regular(3.87),
    eta1_regular(1.84),
    eta0_uniform(1.00),
    eta1_uniform(1.00),
    eta0_sparse(0.97),
    eta1_sparse(6.33),
    eta0_gen(eta0_dense),
    eta1_gen(eta1_dense),

    reportfreq(rfreq),
    epsilon(1e-30),
    logepsilon(log(epsilon)),
    
    tau0(1024),
    nodetau0(1024),
    nodekappa(0.5),
    kappa(0.9),
    online_iterations(50),
    conv_nupdates(1000),
    conv_thresh1(1e-04),
    conv_thresh2(1e-02),

    undirected(true),
    randompair(false),
    randomnode(false),
   
    // bfs
    bfs(false),
    bfs_nodes(10),
    citation(true),
    accuracy(accuracy),
    stopthresh(sth),
    infthresh(inf),
    nonuniform(nu),
    benchmark(bmark),
    randzeros(rzeros),
    preprocess(pre),
    strid(sid),
    groups_file(groups_fname),
    glm(glmopt),
    glm2(glmopt2),
    pcp(pcpopt),
    postprocess(pp),
    acc(accopt),
    lcacc(lcaccopt),
    ngscale(ng),
    link_thresh(lt),
    lt_min_deg(lt_min_deg_opt),
    lowconf(lc),
    nolambda(nol),
    nmemberships(nmem),
    amm(ammopt),
    onesonly(oo),
    terminate(false),

    datdir(dfname),
    deterministic(false),

    //gen
    disjoint(dis),
    gen(gen1),

    //ppc
    ppc_ndraws(100),

    // GAMMA POISSON
    //default_shape(0.001),
    //default_rate(0.001)
    //default_shape(100),
    //default_rate(100),

    default_shape(0.5),
    default_rate(2),

    //default_shape(0.01),
    //default_rate(1),
    nthreads(nthrs),
    label(lbl),
    eta_type(etype),
    nmi(nmival),
    ground_truth_fname(gfname),
    
    use_validation_stop(true),
    use_training_stop(false),
    use_init_communities(init_comm),
    init_communities_fname(init_comm_fname),
    node_scaling_on(nscaling),
    lpmode(lpm),
    gammatrim(gtrim),
    log_training_likelihood(false),
    max_iterations(max_itr),
    globalmu(gmu),
    adagrad(agrad)
{
  assert (!(batch && (strat || rnode || rpair)));

  if (nodelay)
    delaylearn = false;
  else
    delaylearn = true;

  if (strat)
    stratified = true;
  else
    stratified = false;

  if (rnode)
    randomnode = true;

  if (rpair)
    randompair = true;


  //if (massive) {
  //   use_validation_stop = true;
  //   use_training_stop = false;
  // }

  ostringstream sa;
  if (!gen) {
    sa << "n" << n << "-";
    sa << "k" << k;
    if (label != "")
      sa << "-" << label;
    else if (datdir.length() > 3 && 
	     datdir.find("mmsb_gen.dat") == string::npos) {
      string q = datdir.substr(0,2);
      if (q == "..")
	q = "xx";
      sa << "-" << q;
    }
    if (batch)  {
      sa << "-batch";
      reportfreq = 1;
    } else if (single)
      sa << "-sbm";
    else if (massive)
      sa << "-massive";
    else if (acc)
      sa << "-acc";
    else if (lcacc)
      sa << "-lcacc";
    else if (amm)
      sa << "-amm";
    else { 
      if (stratified || delaylearn || undirected || randomnode)
	sa << "-";
      if (subsample_scale > 1)
	sa << "scale" << subsample_scale << "-";
      if (stratified)
	sa << "S";
      if (delaylearn)
	sa << "U";
      if (randompair)
	sa << "rpair";
      if (randomnode)
	sa << "rnode";
      if (nonuniform)
	sa << "R";
    }

    if (glm)
      sa << "glm";
    if (glm2)
      sa << "glm2";

    if (adagrad)
      sa << "-agrad";

    if (globalmu)
      sa << "-gmu";

    if (pcp)
      sa << "pcp";

    if (lowconf)
      sa << "-lowconf";
    if (onesonly)
      sa << "-onesonly";

    if (node_scaling_on)
      sa << "-node-scaling-on";
    
    if (lpmode)
      sa << "-lpmode";
    if (gammatrim)
      sa << "-gtrim";

    if (fastinit)
      sa << "fastinit";

    if (init_comm)
      sa << "-initcomm";

    if (ngscale > 1)
      sa << "-ngscale";

    if (nolambda)
      sa <<"-nolambda";

    if ((gammatrim || fastinit) && nmemberships > 0)
      sa << "-v" << nmemberships;

    if (gap)
      sa << "-GAP";
    if (nthreads > 0)
      sa << "-T" << nthreads;
    if (itype > 0)
      sa << "-i" << itype;
    if (eta_type != "default")
      sa << "-" << eta_type;
  } else {
    assert (!sbm);
    if (disjoint)
      sa << "gend-";
    else
      sa << "gen-";
    sa << "n" << n << "-";
    sa << "k" << k << "-";
    if (eta0_gen == eta0_sparse)
      sa << "sparse";
    else if (eta0_gen == eta0_dense)
      sa << "dense";
    else
      sa << "regular";
  }

  prefix = sa.str();
  level = Logger::DEBUG;

  if (!ppc) {
    fprintf(stdout, "+ Creating directory %s\n", prefix.c_str());
    assert (Logger::initialize(prefix, "infer.log", 
			       force_overwrite_dir, level) >= 0);
    fflush(stdout);
    _plogf = fopen(file_str("/param.txt").c_str(), "w");
    if (!_plogf)  {
      printf("cannot open param file:%s\n",  strerror(errno));
      exit(-1);
    }
    plog("nodes", n);
    plog("groups", k);
    plog("t", t);
    plog("minibatch", s);
    plog("mbsize", m);
    plog("alpha", alpha);
    plog("ngscale", ngscale);
    plog("sbm_alpha", alpha);
    plog("heldout_ratio", heldout_ratio);
    plog("precision_ratio", precision_ratio);
    plog("stratified", stratified);
    plog("nmemberships", nmemberships);
    plog("delaylearn", delaylearn);
    plog("nolambda", nolambda);
    plog("randomnode", randomnode);
    plog("gen", gen);
    plog("undirected", undirected);
    plog("gap", gap);
    plog("nthreads", nthreads);
    plog("disjoint", disjoint);
    plog("stopthresh", stopthresh);
    plog("infthresh", infthresh);
    plog("randzeros", randzeros);
    plog("benchmark", benchmark);
    plog("link_thresh", link_thresh);
    plog("lt_min_deg", lt_min_deg);
    plog("lowconf", lowconf);
    plog("onesonly", onesonly);
    plog("use_init_communities", use_init_communities);
    plog("use_valindation_stop", use_validation_stop);
    plog("node_scaling_on", node_scaling_on);
    plog("lpmode", lpmode);
    plog("gammatrim", gtrim);
    plog("glm", glm);
    plog("glm2", glm2);
    plog("pcp", pcp);
    plog("postprocess", postprocess);
    plog("max_iterations", max_iterations);
    
    //plog("conv_nupdates", conv_nupdates);
    //plog("conv_thresh1", conv_thresh1);
    //plog("conv_thresh2", conv_thresh2);
  }
  unlink(file_str("/mutual.txt").c_str());
  fprintf(stderr, "+ done initializing env\n");
}

/* 
   src: http://www.delorie.com/gnu/docs/glibc/libc_428.html
   Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  
*/
inline int
timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

inline void
timeval_add (struct timeval *result, const struct timeval *x)
{
  result->tv_sec  += x->tv_sec;
  result->tv_usec += x->tv_usec;
  
  if (result->tv_usec >= 1000000) {
    result->tv_sec++;
    result->tv_usec -= 1000000;
  }
}

#endif
