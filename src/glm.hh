#ifndef GLM_HH
#define GLM_HH

#include <list>
#include <utility>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>

#include "env.hh"
#include "matrix.hh"
#include "network.hh"
#include "thread.hh"
#include "tsqueue.hh"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

#define SQ(v) (v * v)
#define CB(v) (v * v * v)

//#define GLOBAL_MU 1

class GLMNetwork;
class LocalCompute {
public:
  LocalCompute(const Env &env, GLMNetwork &glm);
  ~LocalCompute() { }

  bool valid() const { return _valid; }
  void reset(uint32_t p, uint32_t q, yval_t y);

  const Matrix& phi() const { return _phi; }
  void update_phi();
  
  void compute_X_and_XS(uint32_t a, uint32_t b);
  double cached_log_X() const;
  double cached_log_XS() const;

private:
  const Env &_env;
  GLMNetwork &_glm;

  uint32_t _n;
  uint32_t _k;
  uint32_t _t;

  uint32_t _p;
  uint32_t _q;
  uint32_t _y;

  Matrix _phi;
  bool _valid;
  double _log_X;
  double _log_XS;
};

class GLMNetwork {
public:
  GLMNetwork(Env &env, Network &network);
  ~GLMNetwork();

  void gen();
  void randomnode_infer();
  void randompair_infer();
  void randompair_infer_opt();
  void informative_sampling_infer();
  void link_infer();
  void batch_infer();
  void infer();
  
  void set_dir_exp(const Matrix &u, Matrix &exp);
  void set_dir_exp(uint32_t a, const Matrix &u, Matrix &exp);

private:
  void init_gamma();
  void estimate_pi();
  void assign_training_links();
  void shuffle_nodes();
  void write_sample(FILE *f, SampleMap &mp);

  void opt_process(vector<uint32_t> &nodes, uint32_t &links, uint32_t &nonlinks,
		   uint32_t &start_node);
  void opt_process_noninf(vector<uint32_t> &nodes, uint32_t &links, uint32_t &nonlinks,
			  uint32_t &start_node);

  void save_groups();
  void save_model();
  void save_popularity();
  void save_mu();
  double approx_log_likelihood();
  uint32_t duration() const;

  void process(uint32_t p, uint32_t q, double scale = 1.0);

  void estimate_pi(uint32_t p, Array &pi_p) const;
  double pair_likelihood(uint32_t p, uint32_t q, yval_t y) const;
  double pair_likelihood2(uint32_t p, uint32_t q, yval_t y) const;
  string edgelist_s(EdgeList &elist);
  void set_heldout_sample(int s);
  void set_heldout_sample2(int s);
  void set_heldout_degrees();
  void set_validation_sample(int s);
  void set_training_sample(int s);
  void init_heldout();
  void load_heldout_sets();

  double heldout_likelihood(bool nostop=false);
  double precision_likelihood(bool nostop=false);
  double link_prob(uint32_t p, uint32_t q, double &a1, double &a2,
		   double &a3, double &a4) const;
  void write_rank();
  void write_ranking_file();
  double validation_likelihood();
  double training_likelihood();
  int load_gamma();
  int load_only_gamma();
  void load_nodes_for_precision();

  void get_random_edge(bool link, Edge &e) const;
  bool edge_ok(const Edge &e) const;
  void compute_and_log_groups();
  void write_communities(MapVec &communities, string name);
  void write_nodemap(FILE *f, NodeMap &mp);
  void compute_mutual(string s);
  double find_max_k(uint32_t i, uint32_t j, 
		    Array &pi_i, Array &pi_j, uint32_t &max_k);

  yval_t get_y(uint32_t p, uint32_t q);
  uint32_t most_likely_group(uint32_t p);
  void do_on_stop();

  Env &_env;
  Network &_network;

  uint32_t _n;
  uint32_t _k;
  uint32_t _t;

  double _ones_prob;
  double _zeros_prob;
  uint32_t _total_pairs;

  Array _alpha;
  Array _beta;
  double _epsilon;

  Matrix _pi;
  Array _theta;

  // hyperparameters
  double _mu0; // beta
  double _sigma0;

  double _mu1; // theta
  double _sigma1;

  uint32_t _ones;
  AdjMatrix _y;

  Matrix _gamma;
  Matrix _gammat;

  Array _lambda;
  double _sigma_theta;

  Array _lambdat;
  double _sigma_thetat;

  Array _mu;
  double _globalmu;
  double _globalmut;
  double _sigma_beta;

  Array _mut;
  double _sigma_betat;

  Matrix _Elogpi;

  double _rho;
  double _tau0;
  double _kappa;
  double _murho;
  double _mutau0;
  double _mukappa;

  Array _noderhot;
  Array _nodec;
  
  uint32_t _iter;
  uint32_t _start_node;
  
  LocalCompute _lc;
  BoolMap _nodes;

  time_t _start_time;
  FILE *_lf;
  FILE *_vf;
  FILE *_pf;
  FILE *_hf;
  FILE *_trf;
  FILE *_hef;
  FILE *_pef;
  FILE *_vef;
  FILE *_tef;

  SampleMap _heldout_map;
  SampleMap _precision_map;
  SampleMap _validation_map;
  SampleMap _training_map;
  IDMap _heldout_deg;

  EdgeList _heldout_pairs;
  EdgeList _precision_pairs;
  EdgeList _validation_pairs;
  EdgeList _training_pairs;
  NodeMap _sampled_nodes;

  mutable uint32_t _nh;
  double _prev_h, _max_h;

  gsl_rng *_r;
  friend class LocalCompute;

  MapVec _communities;  
  MapVec _communities2;  
  MapVec _communities3;  
  MapVec _communities4;  

  DoubleMap _degstats;
  IDMap _ndegstats;

  Matrix _links;
  uint32_t _nlinks;
  Array _training_links;

  double _inf_epsilon;
  uint32_t _noninf_setsize;
  uArray _shuffled_nodes;

  ValueMap _vmap;
  CountMap _nvmap;
  uArray _ignore_npairs;
  bool _save_ranking_file;
};

//
// local step
//

inline
LocalCompute::LocalCompute(const Env &env, GLMNetwork &glm)
  :_env(env), _glm(glm), 
   _n(_glm._n), _k(_glm._k), _t(_glm._t),
   _phi(_k,_k), 
   _valid(false),
   _log_X(1.0),_log_XS(1.0)
{ 
}

inline void
LocalCompute::reset(uint32_t p, uint32_t q, yval_t y)
{
  _p = p;
  _q = q;
  _y = y;
  _valid = false;
  _log_X = 1.0;
  _log_XS = 1.0;
}

inline void
LocalCompute::compute_X_and_XS(uint32_t a, uint32_t b)
{
  const Array &lambda = _glm._lambda;
  const double &sigma_theta = _glm._sigma_theta;
  const double &sigma_beta = _glm._sigma_beta;
  const double &epsilon = _glm._epsilon;
  const Array &mu = _glm._mu;
  double globalmu = _glm._globalmu;

  double **phid = _phi.data();

  double r1 = lambda[a] + SQ(sigma_theta) + lambda[b];
  Array list_of_exps(_k+2);
  Array list_of_exps2(_k+1);

  list_of_exps[0] = .0;
  double s = .0;

  for (uint32_t k = 0; k < _k; ++k) {
    double l = phid[k][k];
    s += l;
    if (l < 1e-30)
      l = 1e-30;
    
#ifdef GLOBAL_MU
    double u2 = log(l) + lambda[a] + lambda[b] +	\
      SQ(sigma_theta) + globalmu + SQ(sigma_beta)/2;
#else
    double u2 = log(l) + lambda[a] + lambda[b] +       
      SQ(sigma_theta) + mu[k] + SQ(sigma_beta)/2;
#endif
    
    list_of_exps[k+1]=u2;
    list_of_exps2[k]=u2;
  }
  double ss = (1 - s);
  if (ss < 1e-30)
    ss = 1e-30;
  double u3 = log(ss) + lambda[a] + lambda[b] +	\
    SQ(sigma_theta) + epsilon;

  list_of_exps[_k+1] = u3;
  list_of_exps2[_k] = u3;

  tst("r2:%s\n", list_of_exps.s().c_str());
  double r2 = list_of_exps.logsum();
  tst("r3:%s\n", list_of_exps2.s().c_str());
  double r3 = list_of_exps2.logsum();

  tst("r1=%f, r2=%f, r3=%f\n", r1, r2, r3);
  tst("mu=%s\n", _glm._mu.s().c_str());
  tst("lambda[%d] = %f, lambda[%d] = %f\n", a, b, 
      lambda[a], lambda[b]);
  
  _log_X = r1 - r2;
  _log_XS = r3 - r2;

  //tst("(%d:%d) _log_X = %f, _log_XS = %f\n", a, b, _log_X, _log_XS);
  //tst("(%d:%d) _X = %f, _XS = %f\n", a, b, exp(_log_X), exp(_log_XS));
}

#if 0
inline double
LocalCompute::X2(uint32_t a, uint32_t b)
{
  const Array &lambda = _glm._lambda;
  const double &sigma_theta = _glm._sigma_theta;

  double v = exp(lambda[a] + SQ(sigma_theta)/2) * \
    exp(lambda[b] + SQ(sigma_theta)/2);
  double s = S(a,b);
  //tst("v=%f\ts=%f\n", v, s);
  _X = v / (1 + v * s);
  tst("(%d:%d) _X = %f\n", a, b, _X);
  return _X;
}
#endif

inline double
LocalCompute::cached_log_X() const
{
  assert (valid());
  return _log_X;
}

inline double
LocalCompute::cached_log_XS() const
{
  assert (valid());
  return _log_XS;
}

//
// GLM network
//

inline void
GLMNetwork::set_dir_exp(const Matrix &u, Matrix &exp)
{
  const double ** const d = u.data();
  double **e = exp.data();
  for (uint32_t i = 0; i < u.m(); ++i) {
    // psi(e[i][j]) - psi(sum(e[i]))
    double s = .0;
    for (uint32_t j = 0; j < u.n(); ++j) 
      s += d[i][j];
    fflush(stdout);
    assert (s > .0);
    double psi_sum = gsl_sf_psi(s);
    for (uint32_t j = 0; j < u.n(); ++j)
      e[i][j] = gsl_sf_psi(d[i][j]) - psi_sum;
  }
}

inline void
GLMNetwork::set_dir_exp(uint32_t a, const Matrix &u, Matrix &exp)
{
  const double ** const d = u.data();
  double **e = exp.data();

  double s = .0;
  for (uint32_t j = 0; j < u.n(); ++j) 
    s += d[a][j];
  double psi_sum = gsl_sf_psi(s);
  for (uint32_t j = 0; j < u.n(); ++j) 
    e[a][j] = gsl_sf_psi(d[a][j]) - psi_sum;
}

inline uint32_t
GLMNetwork::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
GLMNetwork::edge_ok(const Edge &e) const
{
  if (e.first == e.second)
    return false;
  
  const SampleMap::const_iterator u = _heldout_map.find(e);
  if (u != _heldout_map.end())
    return false;

  const SampleMap::const_iterator w = _validation_map.find(e);
  if (w != _validation_map.end())
    return false;

  const SampleMap::const_iterator v = _precision_map.find(e);
  if (v != _precision_map.end()) {
    return false;
  }
  return true;
}

inline yval_t
GLMNetwork::get_y(uint32_t p, uint32_t q)
{
  return _network.y(p,q);
}


inline uint32_t
GLMNetwork::most_likely_group(uint32_t p)
{
  const double **pid = _pi.const_data();
  double max_k = .0, max_p = .0;
  
  for (uint32_t k = 0; k < _k; ++k)
    if (pid[p][k] > max_p) {
      max_p = pid[p][k];
      max_k = k;
    }
  return max_k;
}


#endif
