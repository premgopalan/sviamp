#include "glm.hh"
#include "log.hh"
#include <sys/time.h>
#include <math.h>

GLMNetwork::GLMNetwork(Env &env, Network &network)
  : _env(env), _network(network),
    _n(env.n), _k(env.k),
    _t(env.t), _alpha(_k), _beta(_k), 
    _epsilon(0),
    _pi(_n,_k), _theta(_n),
    _mu0(0.0), _sigma0(1.0),
    _mu1(0.0), _sigma1(10.0),
    _ones(0), _y(_n,_n),
    _gamma(_n,_k), _gammat(_n,_k), 
    _lambda(_n),
    _sigma_theta(0.1),
    _lambdat(_n), _sigma_thetat(.0),
    _mu(_k),
    _globalmu(.0), _globalmut(.0),
    _sigma_beta(0.5),
    _mut(_k), _sigma_betat(.0),
    _Elogpi(_n,_k), 
    //    _rho(.0), _tau0(65536*2), _kappa(0.9), 
    _rho(.0), _tau0(65536), _kappa(0.5), 
    _murho(.0), _mutau0(65536*2), _mukappa(0.9),
    _noderhot(_n), _nodec(_n),
    _start_node(0),
    _lc(env, *this),
    _nh(0), _prev_h(-2147483647), 
    _max_h(-2147483647),
    _links(1000000,2),
    _nlinks(0), _training_links(_n),
    _inf_epsilon(0.5), 
    //_inf_epsilon(0.5),
    _noninf_setsize(100),
    _shuffled_nodes(_n)
{
  if (!_env.onesonly)
    _inf_epsilon = 0.01;

  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);

  _alpha.set_elements(env.alpha);
  for (uint32_t k = 0; k < _k; ++k)
    _beta[k] = _mu0 +  gsl_ran_gaussian(_r, _sigma0);

  _total_pairs = _n * (_n - 1) / 2;
  _ones_prob = double(_network.ones()) / _total_pairs;
  _zeros_prob = 1 - _ones_prob;

  double n0, n1;
  n0 = _total_pairs * _ones_prob / _k;
  n1 = (_total_pairs / (_k * _k)) - _env.eta0;
  if (n1 < 0)
    n1 = 1.0;

  double p = n0 / (n0 + n1);
  Env::plog("inference n", _n);
  Env::plog("inference links", n0);
  Env::plog("inference non-links", n1);
  Env::plog("expected prob of links within community", p);

  _mu1 = .0; //log (p) - log (1 - p);
  _mu.zero();
  for (uint32_t k = 0; k < _k; ++k) 
    _mu[k] = .0;
  //_mu[k] = _mu0 + gsl_ran_gaussian(_r, 0.1);

  _lambda.zero();

  Env::plog("init mu0", _mu0);
  Env::plog("init mu", _mu);

  Env::plog("network ones", _network.ones());
  Env::plog("network singles", _network.singles());

  Env::plog("init sigma_theta", _sigma_theta);
  Env::plog("init sigma_beta", _sigma_beta);
  Env::plog("(theta prior) mu1", _mu1);
  Env::plog("(theta prior) sigma1", _sigma1);
  Env::plog("(beta prior) mu0", _mu0);
  Env::plog("(beta prior) sigma0", _sigma0);
  Env::plog("epsilon", _epsilon);
  Env::plog("ones_prob", _ones_prob);
  Env::plog("zeros_prob", _zeros_prob);
  Env::plog("mukappa", _mukappa);
  Env::plog("mutau0", _mutau0);
  Env::plog("kappa", _kappa);
  Env::plog("tau0", _tau0);

  _hef = fopen(Env::file_str("/heldout-pairs.txt").c_str(), "w");
  if (!_hef)  {
    lerr("cannot open heldout pairs file:%s\n",  strerror(errno));
    exit(-1);
  }

  _pef = fopen(Env::file_str("/precision-pairs.txt").c_str(), "w");
  if (!_pef)  {
    lerr("cannot open precision pairs file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vef = fopen(Env::file_str("/validation-pairs.txt").c_str(), "w");
  if (!_vef)  {
    lerr("cannot open validation edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  if (_env.log_training_likelihood) {
    _tef = fopen(Env::file_str("/training-pairs.txt").c_str(), "w");
    if (!_tef)  {
      lerr("cannot open training edges file:%s\n",  strerror(errno));
      exit(-1);
    }
    _trf = fopen(Env::file_str("/training.txt").c_str(), "w");
    if (!_trf)  {
      lerr("cannot open training file:%s\n",  strerror(errno));
      exit(-1);
    }
  }

  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }

  _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
  if (!_hf)  {
    lerr("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    lerr("cannot open validation file:%s\n",  strerror(errno));
    exit(-1);
  }


  if (_env.model_load)  {
    if (!_env.amm)
      assert(load_gamma() >= 0);
    else {
      assert(load_only_gamma() >= 0);
      init_heldout();
      //estimate_pi();
      //compute_and_log_groups();
    }
  } else {
    init_gamma();
    load_heldout_sets();
    //init_heldout();
  }

  uint32_t max;
  double avg;
  _network.deg_stats(max, avg);
  Env::plog("avg degree", avg);
  Env::plog("max degree", max);

  if (!_env.nolambda) {
    // XXXXX
    //for (uint32_t n = 0; n < _n; ++n)
    //_lambda[n] = _mu1 + gsl_ran_gaussian(_r, 0.001);

    for (uint32_t n = 0; n < _n; ++n) {
      uint32_t d = _network.deg(n);
      _lambda[n] = log((double)d/max + 1e-5);
    }
  }
  
  shuffle_nodes();
  _start_time = time(0);
  //approx_log_likelihood();
  heldout_likelihood();
  validation_likelihood();
  if (_env.log_training_likelihood)
    training_likelihood();
  set_dir_exp(_gamma, _Elogpi);
}

int
GLMNetwork::load_only_gamma()
{
  fprintf(stderr, "+ loading gamma\n");
  double **gd = _gamma.data();
  FILE *gammaf = fopen("gamma.txt", "r");
  if (!gammaf)
    return -1;
  uint32_t n = 0;
  int sz = 32*_k;
  char *line = (char *)malloc(sz);
  while (!feof(gammaf)) {
    if (fgets(line, sz, gammaf) == NULL)
      break;
    //assert (fscanf(gammaf, "%[^\n]", line) > 0);
    debug("line = %s\n", line);
    uint32_t k = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing gamma file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 2) // skip node id and seq
	gd[n][k-2] = d;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, sz);
  }
  assert (n == _n);
  fclose(gammaf);
  return 0;
}

int
GLMNetwork::load_gamma()
{
  fprintf(stderr, "+ loading gamma\n");
  double **gd = _gamma.data();
  FILE *gammaf = fopen("gamma.txt", "r");
  if (!gammaf)
    return -1;
  uint32_t n = 0;
  int sz = 32*_k;
  char *line = (char *)malloc(sz);
  while (!feof(gammaf)) {
    if (fgets(line, sz, gammaf) == NULL)
      break;
    //assert (fscanf(gammaf, "%[^\n]", line) > 0);
    debug("line = %s\n", line);
    uint32_t k = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing gamma file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 2) // skip node id and seq
	gd[n][k-2] = d;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, sz);
  }
  assert (n == _n);

  fclose(gammaf);
  load_heldout_sets();
}

void
GLMNetwork::load_heldout_sets()
{
  uint32_t n = 0;
  uint32_t a, b;
  const IDMap &id2seq = _network.id2seq();
  FILE *f = fopen("heldout-edges.txt", "r");
  while (!feof(f)) {
    if (fscanf(f, "%d\t%d\t%*s\n", &a, &b) < 0) {
      fprintf(stderr, "error: cannot read heldout file\n");
      exit(-1);
    }
    
    IDMap::const_iterator i1 = id2seq.find(a);
    IDMap::const_iterator i2 = id2seq.find(b);
    
    if ((i1 == id2seq.end()) || (i2 == id2seq.end())) {
      fprintf(stderr, "error: id %d or id %d not found in original network\n", 
	      a, b);
      exit(-1);
    }
    Edge e(i1->second,i2->second);
    Network::order_edge(_env, e);
    _heldout_pairs.push_back(e);
    _heldout_map[e] = true;
    ++n;
  }
  Env::plog("loaded heldout pairs:", n);
  fprintf(_hef, "%s\n", edgelist_s(_heldout_pairs).c_str());
  fclose(_hef);
  fclose(f);

  n = 0;
  f = fopen("precision-edges.txt", "r");
  while (!feof(f)) {
    if (fscanf(f, "%d\t%d\t%*s\n", &a, &b) < 0) {
      fprintf(stderr, "error: cannot read precision file\n");
      exit(-1);
    }
    
    IDMap::const_iterator i1 = id2seq.find(a);
    IDMap::const_iterator i2 = id2seq.find(b);
    
    if ((i1 == id2seq.end()) || (i2 == id2seq.end())) {
      fprintf(stderr, "error: id %d or id %d not found in original network\n", 
	      a, b);
      exit(-1);
    }
    Edge e(i1->second,i2->second);
    Network::order_edge(_env, e);
    _precision_pairs.push_back(e);
    _precision_map[e] = true;
    ++n;
  }
  Env::plog("loaded precision pairs:", n);
  fprintf(_pef, "%s\n", edgelist_s(_precision_pairs).c_str());
  fclose(_pef);
  fclose(f);
}

void
GLMNetwork::shuffle_nodes()
{
  for (uint32_t i = 0; i < _n; ++i)
    _shuffled_nodes[i] = i;
  gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
}

GLMNetwork::~GLMNetwork()
{
  fclose(_lf);
  fclose(_hf);
  fclose(_vf);
  if (_env.log_training_likelihood)
    fclose(_trf);
}

string
GLMNetwork::edgelist_s(EdgeList &elist)
{
  ostringstream sa;
  for (EdgeList::const_iterator i = elist.begin(); i != elist.end(); ++i) {
    const Edge &p = *i;
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator a = m.find(p.first);
    IDMap::const_iterator b = m.find(p.second);
    yval_t y = _network.y(p.first, p.second);
    if (a != m.end() && b!= m.end()) {
      sa << a->second << "\t" << b->second << "\t" << (int)y << "\n";
    }
  }
  return sa.str();
}

void
GLMNetwork::set_heldout_sample(int s)
{
  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p)
      get_random_edge(true, e); // link
    else
      get_random_edge(false, e); // nonlink

    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

    if (y == 0 and c0 < p) {
      c0++;
      _heldout_pairs.push_back(e);
      _heldout_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _heldout_pairs.push_back(e);
      _heldout_map[e] = true;
    }
  }
}


void
GLMNetwork::set_heldout_sample2(int s)
{
  int ones = 0, zeros = 0;
  while (ones < s / 2) {
    Edge e;
    get_random_edge(true, e); // link
    ones++;
    _heldout_pairs.push_back(e);
    _heldout_map[e] = true;

    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

    uint32_t mm = 0;
    uint32_t limit = 1;
    const vector<uint32_t> *v = _network.get_edges(a);
    for (uint32_t j = 0; j < v->size() && mm < limit; ++j) {    
      uint32_t q = (*v)[j];
      const vector<uint32_t> *u = _network.get_edges(q);
      for (uint32_t k = 0; u && k < u->size() && mm < limit; ++k) {
	uint32_t c = (*u)[k];
	if (a != c && _network.y(a,c) == 0) {
	  Edge f(a,c);
	  _heldout_pairs.push_back(f);
	  _heldout_map[f] = true;
	  mm++;
	  zeros++;
	}
      }
    }
  }
  Env::plog("heldout ones:", ones);
  Env::plog("heldout zeros:", zeros);
}


void
GLMNetwork::set_heldout_degrees()
{
  for (SampleMap::const_iterator i = _heldout_map.begin();
       i != _heldout_map.end(); ++i) {
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    _heldout_deg[p]++;
    _heldout_deg[q]++;
  }
}

void
GLMNetwork::set_validation_sample(int s)
{
  if (_env.accuracy)
    return;
  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p)
      get_random_edge(true, e); // link
    else
      get_random_edge(false, e); // nonlink

    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

    if (y == 0 and c0 < p) {
      c0++;
      _validation_pairs.push_back(e);
      _validation_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _validation_pairs.push_back(e);
      _validation_map[e] = true;
    }
  }
}


void
GLMNetwork::set_training_sample(int s)
{
  if (!_env.log_training_likelihood)
    return;

  int c0 = 0;
  int c1 = 0;
  int p = s / 2;
  while (c0 < p || c1 < p) {
    Edge e;
    if (c0 == p)
      get_random_edge(true, e); // link
    else
      get_random_edge(false, e); // nonlink

    uint32_t a = e.first;
    uint32_t b = e.second;
    yval_t y = get_y(a,b);

    if (y == 0 and c0 < p) {
      c0++;
      _training_pairs.push_back(e);
      _training_map[e] = true;
    }
    if (y == 1 and c1 < p) {
      c1++;
      _training_pairs.push_back(e);
      _training_map[e] = true;
    }
  }
}


void
GLMNetwork::init_gamma()
{
  double **d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _k; ++j)  {
      double v = (_k < 100) ? 1.0 : (double)100.0 / _k;
      d[i][j] = gsl_ran_gamma(_r, 100 * v, 0.01);
    }
}

void
GLMNetwork::gen()
{
  double *alphad = _alpha.data();
  yval_t **yd = _y.data();
  double **pid = _pi.data();
  
  for (uint32_t i = 0; i < _n; ++i) {
    // draw pi from alpha
    gsl_ran_dirichlet(_r, _k, alphad, pid[i]);
    
    // draw theta from a Gaussian
    _theta[i] = _mu1 + gsl_ran_gaussian(_r, _sigma1);
  }

  debug("pi = %s", _pi.s().c_str());
  debug("theta = %s", _theta.s().c_str());
  debug("beta = %s", _beta.s().c_str());
  
  uArray zi(_k), zj(_k);
  uint32_t *zid = zi.data();
  uint32_t *zjd = zj.data();
  for (uint32_t i = 0; i < _n; ++i) 
    for (uint32_t j = 0; j < _n; ++j)  {
      if (i >= j)
	continue;

      gsl_ran_multinomial(_r, _k, 1, pid[i], zid);
      gsl_ran_multinomial(_r, _k, 1, pid[j], zjd);

      debug("zi = %s\n", zi.s().c_str());
      debug("zj = %s\n", zj.s().c_str());

      uint32_t idx = (uint32_t)dot(zi, zj);
      double logodds = .0;
      if (idx == 0) {
	debug("epsilon\n");
	logodds = 0;
      } else {
	for (uint32_t k = 0; k < _k; ++k) 
	  if (zi[k]) {
	    debug("idx = %d\n", k);
	    logodds = _beta[k];
	  }
      }
      logodds += _theta[i] + _theta[j];

      debug("logodds = %.5f\n", logodds);
      // apply logit-inverse function
      double l = 1.0 / (1 + exp(-logodds));
      yd[i][j] = gsl_ran_bernoulli(_r, l);
      debug("E[y|.] = %.5f\n", l);
      debug("yd[i][j] = %d\n", yd[i][j]);
      if (yd[i][j])
	_ones++;
    }
  debug("y = %s", _y.s().c_str());
}

void
LocalCompute::update_phi()
{
  const Array &mu = _glm._mu;
  double globalmu = _glm._globalmu;
  const Matrix &Elogpi = _glm._Elogpi;
  const double &sigma_beta = _glm._sigma_beta;
  const double &epsilon = _glm._epsilon;
  double **phid = _phi.data();
  const double ** const elogpid = _glm._Elogpi.const_data();

  compute_X_and_XS(_p,_q);
  for (uint32_t k = 0; k < _k; ++k) { 
#ifdef GLOBAL_MU    
    double u1 = _log_X + globalmu + SQ(sigma_beta)/2;
#else
    double u1 = _log_X + mu[k] + SQ(sigma_beta)/2;
#endif
    double u2 = _log_X + epsilon;
    double u = exp(u1) - exp(u2);
    
    phid[k][k] = Elogpi.at(_p,k) + Elogpi.at(_q,k);
#ifdef GLOBAL_MU
    phid[k][k] += (_y * (globalmu - epsilon) - u);
#else
    phid[k][k] += (_y * (mu[k] - epsilon) - u);
#endif
    debug("Elogpi(%d,%d) = %f\n", _p, k, Elogpi.at(_p,k));
  }
  for (uint32_t k1 = 0; k1 < _k ; ++k1)
    for (uint32_t k2 = 0; k2 < _k ; ++k2) 
      if (k1 != k2)  
	phid[k1][k2] = elogpid[_p][k1] + elogpid[_q][k2];
  _phi.lognormalize();
  compute_X_and_XS(_p,_q);
  _valid = true;
}

void
GLMNetwork::init_heldout()
{
  int s = _env.heldout_ratio * _network.ones();
  set_heldout_sample(s);
  set_heldout_degrees();
  //set_validation_sample(s);

#ifdef TRAINING_SAMPLE
  set_training_sample(2*(_network.ones() - s));
#endif

  Env::plog("heldout ratio", _env.heldout_ratio);
  Env::plog("heldout edges (1s and 0s)", _heldout_map.size());
  fprintf(_hef, "%s\n", edgelist_s(_heldout_pairs).c_str());
  fprintf(_vef, "%s\n", edgelist_s(_validation_pairs).c_str());

#ifdef TRAINING_SAMPLE
  fprintf(_tef, "%s\n", edgelist_s(_training_pairs).c_str());
  fclose(_tef);
#endif

  fclose(_hef);
  fclose(_vef);
}

void
GLMNetwork::randomnode_infer()
{
  set_dir_exp(_gamma, _Elogpi);
  while (1) {
    //
    // L step
    //
    NodeMap sampled_nodes;
    vector<uint32_t> nodes;
    do {
      uint32_t start_node = gsl_rng_uniform_int(_r, _n);
      NodeMap::const_iterator itr = sampled_nodes.find(start_node);
      if (itr == sampled_nodes.end()) {
	set_dir_exp(start_node, _gamma, _Elogpi);
	_gammat.zero(start_node);
	nodes.push_back(start_node);
	sampled_nodes[start_node] = true;
      }
    } while (sampled_nodes.size() < _env.sets_mini_batch);
    
    map<uint32_t, bool> neighbor_nodes;
    
    _mut.zero();
    _globalmut = .0;
    _sigma_betat = .0;
    _sigma_thetat = .0;
    _lambdat.zero();

    uint32_t c = 0;
    for (NodeMap::const_iterator itr = sampled_nodes.begin(); 
	 itr != sampled_nodes.end(); ++itr) {
      uint32_t start_node = itr->first;

      const vector<uint32_t> *edges = _network.get_edges(start_node);
      if (!edges)
	continue;
      
      printf("start node = %d, %ld links", start_node, edges->size());
      fflush(stdout);
      
      for (uint32_t i = 0; i < edges->size(); ++i) {
	uint32_t a = (*edges)[i];
	
	Edge e(start_node,a);
	Network::order_edge(_env, e);
	if (!edge_ok(e))
	  continue;
	
	NodeMap::const_iterator nt = neighbor_nodes.find(a);
	NodeMap::const_iterator st = sampled_nodes.find(a);
	if (nt == neighbor_nodes.end() && st == sampled_nodes.end()) {
	  neighbor_nodes[a] = true;
	  set_dir_exp(a, _gamma, _Elogpi);
	  _gammat.zero(a);
	  nodes.push_back(a);
	}
	uint32_t p = e.first;
	uint32_t q = e.second;
	process(p,q);
      }
      
      vector<Edge> sample;
      double v = (double)(gsl_rng_uniform_int(_r, _n)) / _noninf_setsize;
      uint32_t q = ((int)v) * _noninf_setsize;
      tst("\nq = %d, set size = %d\n", q, _noninf_setsize);
      
      while (sample.size() < _noninf_setsize) {
	uint32_t node = _shuffled_nodes[q];
	if (node == start_node) {
	  q = (q + 1) % _n;
	  continue;
	}
	
	yval_t y = get_y(start_node, node);
	Edge e(start_node, node);
	Network::order_edge(_env, e);
	if (y == 0 && edge_ok(e))
	  sample.push_back(e);
	q = (q + 1) % _n;
      }
      
      double scale = (_n - _network.deg(start_node)) / sample.size();
      printf("start node = %d, scale = %f, sample size = %ld\n", 
	     start_node, scale, sample.size());
      for (uint32_t i = 0; i < sample.size(); ++i) {
	Edge e = sample[i];
	assert (edge_ok(e));

	uint32_t p = e.first;
	uint32_t q = e.second;

	uint32_t a;
	if (p != start_node)
	  a = p;
	else
	  a = q;

	nodes.push_back(a);
	set_dir_exp(a, _gamma, _Elogpi);
	_gammat.zero(a);
	process(p,q,scale);
      }
    }
      
    info("* links=%d\n", _network.deg(start_node));
    printf("* mut=%s\n", _mut.s().c_str());
    debug("* sigma_thetat=%.5f\n", _sigma_thetat);
    debug("* sigma_betat=%.5f\n", _sigma_betat);    

    double scale = _n / (2 * sampled_nodes.size());
    // mut
    for (uint32_t k = 0; k < _k; ++k) {
      _globalmut += _mut[k];
      _mut[k] = _mut[k] + ((_mu0 - _mu[k]) / SQ(_sigma0)); // XXXXX scaling
    }
    _globalmut += (_mu0 - _globalmu) / SQ(_sigma0);

    for (NodeMap::const_iterator itr = sampled_nodes.begin(); 
	 itr != sampled_nodes.end(); ++itr) {
      uint32_t n = itr->first;
      _lambdat[n] += (_mu1 -_lambda[n]) / SQ(_sigma1);
      for (uint32_t k = 0; k < _k; ++k) {
	_gammat.set(n, k, _gammat.at(n,k));
	_gammat.add(n, k, _alpha[k] - _gamma.at(n,k));
      }

      _rho = pow(_tau0 + _iter, -1 * _kappa);
      _murho = pow(_mutau0 + _iter, -1 * _mukappa);

      for (uint32_t k = 0; k < _k; ++k)
	_gamma.add(n, k, _rho * _gammat.at(n,k));

      if (!_env.nolambda)
	_lambda[n] += _rho * _lambdat[n];
      set_dir_exp(n, _gamma, _Elogpi);
      
      for (uint32_t k = 0; k < _k; ++k) {
	_mu[k] += _murho * _mut[k];
	if (_mu[k] < .0)
	  _mu[k] = .0;
	//	else if (_mu[k] > 3.0)
	//  _mu[k] = 3.0;
      }
      _globalmu += _murho * _globalmut;
      if (_globalmu < .0)
	_globalmu = .0;
    }
    
    debug("%d:GAMMA = %s\n", _iter, _gamma.s().c_str());
    debug("%d:LAMBDA = %s\n", _iter, _lambda.s().c_str());
    debug("%d:mu=%s\n", _iter, _mu.s().c_str());
    tst("%d:sigma_theta = %.5f\n", _iter, _sigma_theta);
    tst("%d:sigma_beta = %.5f\n", _iter, _sigma_beta);

    _iter++;
    printf("\riteration %d\n", _iter);
    fflush(stdout);
    if (_iter % _env.reportfreq == 0) {
      printf("\niteration %d (skipped heldout %d)\n", _iter, c);
      estimate_pi();
      heldout_likelihood();
      compute_and_log_groups();
      do_on_stop();
      if (_env.terminate) {
	compute_and_log_groups();
	do_on_stop();
	_env.terminate = false;
      }
    }
  }
}


void
GLMNetwork::randompair_infer()
{
  set_dir_exp(_gamma, _Elogpi);
  while (1) {

    if (_env.max_iterations && _iter > _env.max_iterations) {
      printf("+ Quitting: reached max iterations.\n");
      Env::plog("maxiterations reached", true);
      _env.terminate = true;
      do_on_stop();
      exit(0);
    }

    //_nodes.clear();
    EdgeList pairs;
    do {
      Edge e;
      do {
	e.first = gsl_rng_uniform_int(_r, _n);
	e.second = gsl_rng_uniform_int(_r, _n);
	Network::order_edge(_env, e);
	assert(e.first == e.second || Network::check_edge_order(e));
      } while (!edge_ok(e));
      pairs.push_back(e);
    } while (pairs.size() <= (double)_n/2);
    
    //
    // L step
    //
    _gammat.zero();

    _mut.zero();
    _sigma_betat = .0;
    _sigma_thetat = .0;
    _lambdat.zero();

    uint32_t c = 0;
    for (EdgeList::const_iterator i = pairs.begin(); i != pairs.end(); ++i) {
      uint32_t p = i->first;
      uint32_t q = i->second;
      
      process(p,q);
    }

    double scale = _total_pairs / pairs.size();
    // mut
    for (uint32_t k = 0; k < _k; ++k)
      _mut[k] = _mut[k] * scale + ((_mu0 - _mu[k]) / SQ(_sigma0)); // XXXXX

    // lambda_n and gammat
    for (uint32_t n = 0; n < _n; ++n) {
      _lambdat[n] += (_mu1 -_lambda[n]) / SQ(_sigma1); // XXXXX
      for (uint32_t k = 0; k < _k; ++k) {
	_gammat.set(n, k, _gammat.at(n,k) * scale);
	_gammat.add(n, k, _alpha[k] - _gamma.at(n,k));
      }
    }

    // G step
    _rho = pow(_tau0 + _iter, -1 * _kappa);
    _murho = pow(_mutau0 + _iter, -1 * _mukappa);
    
    for (uint32_t n = 0; n < _n; ++n) {
      for (uint32_t k = 0; k < _k; ++k)
	_gamma.add(n, k, _rho * _gammat.at(n,k));
      if (!_env.nolambda) {
	//	_lambda[n] += _rho * _lambdat[n]; XXXXX
      }
    }
    set_dir_exp(_gamma, _Elogpi);    

    for (uint32_t k = 0; k < _k; ++k)
      _mu[k] += _murho * _mut[k];

    _iter++;
    printf("\riteration %d", _iter);
    fflush(stdout);
    if (_iter % _env.reportfreq == 0) {
      printf("\niteration %d (skipped heldout %d)\n", _iter, c);
      estimate_pi();
      heldout_likelihood();
      compute_and_log_groups();
      do_on_stop();
      if (_env.terminate) {
	compute_and_log_groups();
	do_on_stop();
	_env.terminate = false;
      }
    }
  }
}

void
GLMNetwork::randompair_infer_opt()
{
  set_dir_exp(_gamma, _Elogpi);

  while (1) {

    if (_env.max_iterations && _iter > _env.max_iterations) {
      printf("+ Quitting: reached max iterations.\n");
      Env::plog("maxiterations reached", true);
      _env.terminate = true;
      do_on_stop();
      exit(0);
    }

    //_nodes.clear();
    EdgeList pairs;
    do {
      Edge e;
      do {
	e.first = gsl_rng_uniform_int(_r, _n);
	e.second = gsl_rng_uniform_int(_r, _n);
	Network::order_edge(_env, e);
	assert(e.first == e.second || Network::check_edge_order(e));
      } while (!edge_ok(e));
      pairs.push_back(e);
    } while (pairs.size() <= (double)_n/2);
    
    //
    // L step
    //
    _gammat.zero();

    _mut.zero();
    _sigma_betat = .0;
    _sigma_thetat = .0;
    _lambdat.zero();

    uint32_t c = 0;
    double s0 = .0, s1 = .0;
    for (EdgeList::const_iterator i = pairs.begin(); i != pairs.end(); ++i) {
      uint32_t p = i->first;
      uint32_t q = i->second;

      process(p,q);

      yval_t y = _network.y(p,q);

      //if (y == 0)
      //s0 += exp(_lc.cached_log_XS());
      //else
      //s1 += exp(_lc.cached_log_XS());
    }
    
    //printf("%d: (%d:%d) log_X=%f, log_XS=%f\n", y, p, q, log_X, log_XS);
    //printf("0: %.3f\n", s0);
    //printf("1: %.3f\n", s1);
    debug("* links=%d\n", _network.deg(_start_node));
    debug("* before: mut=%s\n", _mut.s().c_str());
    debug("* sigma_thetat=%.5f\n", _sigma_thetat);
    debug("* sigma_betat=%.5f\n", _sigma_betat);

    double scale = _total_pairs / pairs.size();
    // mut
    for (uint32_t k = 0; k < _k; ++k)
      _mut[k] = (_mut[k])  + ((_mu0 - _mu[k]) / SQ(_sigma0)); // XXX scale?
    
    // sigma_betat
    _sigma_betat = - (scale * _sigma_betat);
    debug("** sigma_betat=%.5f\n", _sigma_betat);    
    
    double v = (_k / _sigma_beta);
    _sigma_betat += -(_k * _sigma_beta / SQ(_sigma0)) + v;

    debug("*** sigma_betat=%.5f, _sigma_beta=%f, v=%f\n", _sigma_betat, _sigma_beta, SQ(_sigma0));

    // lambda_n and gammat
    for (uint32_t n = 0; n < _n; ++n) {
    //for (BoolMap::const_iterator b = _nodes.begin(); b != _nodes.end(); ++b) {
    //uint32_t n = b->first;
      _lambdat[n] += (_mu1 -_lambda[n]) / SQ(_sigma1);
      for (uint32_t k = 0; k < _k; ++k) {
	_gammat.set(n, k, _gammat.at(n,k) * scale);
	_gammat.add(n, k, _alpha[k] - _gamma.at(n,k));
      }
    }
    
    // sigma_theta
    v = (_n / _sigma_theta);
    _sigma_thetat = - (scale * _sigma_thetat);
    _sigma_thetat += -(_n * _sigma_theta / SQ(_sigma1)) + v;

    debug("* after: mut=%s\n", _mut.s().c_str());
    debug("* sigma_thetat=%.5f\n", _sigma_thetat);
    debug("* sigma_betat=%.5f\n", _sigma_betat);
    debug("* lambdat=%s\n", _lambdat.s().c_str());

    // G step
    _rho = pow(_tau0 + _iter, -1 * _kappa);
    _murho = pow(_mutau0 + _iter, -1 * _mukappa);
    
    for (uint32_t n = 0; n < _n; ++n) {
      //for (BoolMap::const_iterator b = _nodes.begin(); b != _nodes.end(); ++b) {
      //uint32_t n = b->first;
      for (uint32_t k = 0; k < _k; ++k)
	_gamma.add(n, k, _rho * _gammat.at(n,k));
      if (!_env.nolambda)
	_lambda[n] += _rho * _lambdat[n];
    }
    //_sigma_theta += _rho * _sigma_thetat;
    for (uint32_t k = 0; k < _k; ++k)
      _mu[k] += _murho * _mut[k];
    //_sigma_beta += _rho * _sigma_betat;

    debug("%d:GAMMA = %s\n", _iter, _gamma.s().c_str());
    debug("%d:LAMBDA = %s\n", _iter, _lambda.s().c_str());
    debug("%d:mu=%s\n", _iter, _mu.s().c_str());
    tst("%d:sigma_theta = %.5f\n", _iter, _sigma_theta);
    tst("%d:sigma_beta = %.5f\n", _iter, _sigma_beta);

    _iter++;
    printf("\riteration %d", _iter);
    fflush(stdout);
    if (_iter % _env.reportfreq == 0) {
      printf("\niteration %d (skipped heldout %d)\n", _iter, c);
      estimate_pi();
      heldout_likelihood();
      if (_env.terminate) {
	compute_and_log_groups();
	do_on_stop();
	_env.terminate = false;
      }
    }
  }
}



void
GLMNetwork::do_on_stop()
{
  save_model();
  precision_likelihood();
  auc();
  save_groups();
}

void
GLMNetwork::assign_training_links()
{
  _nlinks = 0;
  double **linksd = _links.data();
  for (uint32_t p = 0; p < _n; ++p)  {
    //const vector<uint32_t> *edges = _network.get_edges(p);
    //for (uint32_t r = 0; r < edges->size(); ++r) {
    //uint32_t q = (*edges)[r];
    for (uint32_t q = 0; q < _n; ++q)  {    

      if (p >= q)
	continue;
      
      Edge e(p, q);
      Network::order_edge(_env, e);
      if (!edge_ok(e))
	continue;
      
      linksd[_nlinks][0] = p;
      linksd[_nlinks][1] = q;
      _nlinks++;

      _training_links[p]++;
      _training_links[q]++;
    }
  }
}


void
GLMNetwork::process(uint32_t p, uint32_t q, double scale)
{
  //printf("process %d,%d\n", p,q);
  //fflush(stdout);
  yval_t y = _network.y(p,q);

  _lc.reset(p,q,y);
  _lc.update_phi();
  
  double log_X = _lc.cached_log_X();
  double log_XS = _lc.cached_log_XS();

  const Matrix &phi = _lc.phi();

  double **gtd = _gammat.data();
  Array phi1k(_k), phi2k(_k);
  for (uint32_t k = 0; k < _k; ++k) {
    phi.slice(0, k, phi1k);
    phi.slice(1, k, phi2k);

    gtd[p][k] += scale * phi1k.sum();
    gtd[q][k] += scale * phi2k.sum();
  }
  
  const double ** const phid = _lc.phi().const_data();
  for (uint32_t k = 0; k < _k; ++k) {
#ifdef GLOBAL_MU
    double u1 = log_X + _globalmu + SQ(_sigma_beta)/2; 
#else
    double u1 = log_X + _mu[k] + SQ(_sigma_beta)/2;
#endif
    _mut[k] += scale * phid[k][k] * (y - exp(u1));
    //printf("mut[%d] = %f\n", k, phid[k][k] * (y - exp(u1)));
  }
  
  // sigma_beta gradient
  double u1 = log_X;
  Array list_of_exps(_k);
  for (uint32_t k = 0; k < _k; ++k) {
    double l = phid[k][k];
    if (l < 1e-30)
      l = 1e-30;
    list_of_exps[k] = log(l) + _mu[k];
  }
  double u2 = list_of_exps.logsum();
  double u = exp(u1 + u2);
  _sigma_betat += _sigma_beta * u;
  
  // lambda_a, lambda_b gradients
  double xs = exp(log_XS);
  _lambdat[p] += scale * (y - xs);
  _lambdat[q] += scale * (y - xs);
  
  // sigma_theta gradient
  _sigma_thetat += 2 * _sigma_theta * xs;
}

void
GLMNetwork::infer()
{
  if (_env.randompair)
    randompair_infer();
  else if (_env.randomnode)
    randomnode_infer();
  else if (_env.lpmode)
    batch_infer();
  else if (_env.informative_sampling)
    informative_sampling_infer();
}

void
GLMNetwork::informative_sampling_infer()
{
  while (1) {
    
    uint32_t links = 0, nonlinks = 0;
    vector<uint32_t> nodes;
    uint32_t type = gsl_ran_bernoulli(_r, _inf_epsilon);
    
    uint32_t start_node;
    if (type == 0)
      opt_process(nodes, links, nonlinks, start_node);
    else // non-informative sets
      opt_process_noninf(nodes, links, nonlinks, start_node);
    
    double scale = type == 0 ? 
      (double)_n / 2 : ((double)_n * (double)_n) / (2 * _inf_epsilon * _noninf_setsize);
    
    for (uint32_t k = 0; k < _k; ++k)
      _mut[k] = (_mut[k] * scale)  + ((_mu0 - _mu[k]) / SQ(_sigma0)); // XXX scale?
    
    for (uint32_t i = 0; i < nodes.size(); ++i) {
      
      uint32_t n = nodes[i];
      if (!_env.nolambda)
	_lambdat[n] += (_mu1 -_lambda[n]) / SQ(_sigma1);
      
      //_lambdat[n] += (_mu1 -_lambda[n]) / SQ(_sigma1);
      for (uint32_t k = 0; k < _k; ++k) {
	_gammat.set(n, k, _gammat.at(n,k) * scale);
	_gammat.add(n, k, _alpha[k] - _gamma.at(n,k));
      }
      _nodec[n]++;
    }
    
    // G step
    _rho = pow(_tau0 + _iter, -1 * _kappa);

    for (uint32_t i = 0; i < nodes.size(); ++i) {
      uint32_t n = nodes[i];
      _noderhot[n] = pow(_tau0 + _nodec[n], -1 * _kappa);
      for (uint32_t k = 0; k < _k; ++k)
	_gamma.add(n, k, _noderhot[n] * _gammat.at(n,k));
      if (!_env.nolambda)
	_lambda[n] += _noderhot[n] * _lambdat[n];
      set_dir_exp(n, _gamma, _Elogpi);
    }

    if (_iter % _env.reportfreq == 0) {
      _murho = pow(_mutau0 + _iter % _env.reportfreq, -1 * _mukappa);
      for (uint32_t k = 0; k < _k; ++k)
	_mu[k] += _murho * _mut[k] / _env.reportfreq;
      _mut.zero();
    }

    _iter++;
    printf("\riteration %d (links:%d,nonlinks:%d,nodes:%ld)", 
	   _iter, links, nonlinks, nodes.size());
    fflush(stdout);
    if (_iter % _env.reportfreq == 0) {
      printf("\niteration %d\n", _iter);
      estimate_pi();
      save_model();
      //compute_and_log_groups();
      heldout_likelihood();
    }
  }
}

void
GLMNetwork::opt_process(vector<uint32_t> &nodes, uint32_t &links, uint32_t &nonlinks, 
			uint32_t &start_node)
{
  start_node = gsl_rng_uniform_int(_r, _n);

  set_dir_exp(start_node, _gamma, _Elogpi);
  _gammat.zero(start_node);
  _lambdat[start_node] = 0;
  nodes.push_back(start_node);
  
  const vector<uint32_t> *edges = _network.get_edges(start_node);
  if (!edges)
    return;

  for (uint32_t i = 0; i < edges->size(); ++i) {
    uint32_t a = (*edges)[i];
    
    Edge e(start_node,a);
    Network::order_edge(_env, e);
    if (!edge_ok(e))
      continue;

    set_dir_exp(a, _gamma, _Elogpi);
    _gammat.zero(a);
    _lambdat[a] = 0;
    nodes.push_back(a);
    
    uint32_t p = e.first;
    uint32_t q = e.second;

    assert (p != q);
    yval_t y = get_y(p, q);
    assert (y == 1);

    process(p,q);
    links++;
  }

  vector<uint32_t> *zeros = NULL;
  if (!_env.onesonly)
    zeros = _network.sparse_zeros(start_node);
  
  for (uint32_t i = 0; zeros && i < zeros->size(); ++i) {
    uint32_t a = (*zeros)[i];
    assert (a != start_node);
    
    yval_t y = get_y(start_node, a);
    assert ((y & 0x01) == 0);
    
    Edge e(start_node,a);
    Network::order_edge(_env, e);
    if (!edge_ok(e))
      continue;

    set_dir_exp(a, _gamma, _Elogpi);
    nodes.push_back(a);
    _gammat.zero(a);
    _lambdat[a] = 0;

    uint32_t p = e.first;
    uint32_t q = e.second;

    process(p,q);
    nonlinks++;
  }
}

void
GLMNetwork::opt_process_noninf(vector<uint32_t> &nodes, uint32_t &links, uint32_t &nonlinks,
			       uint32_t &start_node)
{
  //printf("processed %d links, %d nonlinks\n", links, nonlinks);
  //fflush(stdout);
  start_node = gsl_rng_uniform_int(_r, _n);
  set_dir_exp(start_node, _gamma, _Elogpi);
  _gammat.zero(start_node);
  _lambdat[start_node] = 0;
  nodes.push_back(start_node);

  NodeMap inf_map;
  if (!_env.onesonly) {
    const vector<uint32_t> *zeros = _network.sparse_zeros(_start_node);
    if (zeros)
      for (uint32_t i = 0; i < zeros->size(); ++i)
	inf_map[(*zeros)[i]] = true;
  }

  vector<Edge> sample;
  double v = (double)(gsl_rng_uniform_int(_r, _n)) / _noninf_setsize;
  uint32_t q = ((int)v) * _noninf_setsize;
  tst("\nq = %d, set size = %d\n", q, _noninf_setsize);

  while (sample.size() < _noninf_setsize) {
    uint32_t node = _shuffled_nodes[q];
    if (node == _start_node) {
      q = (q + 1) % _n;
      continue;
    }
    
    yval_t y = get_y(_start_node, node);
    Edge e(_start_node, node);
    Network::order_edge(_env, e);
    if (y == 0 && edge_ok(e)) {
      if (_env.onesonly) 
	sample.push_back(e);
      else {
	NodeMap::iterator itr = inf_map.find(node);
	if (!itr->second) {
	  sample.push_back(e);
	  tst("(%d,%d)\n", e.first, e.second);
	}
      }
    }
    q = (q + 1) % _n;
  };

  //printf("noninf sample size = %ld\n", sample.size());
  //fflush(stdout);
  for (uint32_t i = 0; i < sample.size(); ++i) {
    Edge e = sample[i];
    assert (edge_ok(e));

    uint32_t p = e.first;
    uint32_t q = e.second;
    uint32_t a;
    if (p != _start_node)
      a = p;
    else
      a = q;

    set_dir_exp(a, _gamma, _Elogpi);
    nodes.push_back(a);
    _gammat.zero(a);
    _lambdat[a] = 0;

    yval_t y = get_y(p, q);
    assert (y == 0);

    process(p,q);
    nonlinks++;
  }
}

void
GLMNetwork::batch_infer()
{
  while (1) {
    set_dir_exp(_gamma, _Elogpi);
    _gammat.zero();
    _mut.zero();
    _sigma_betat = .0;
    _sigma_thetat = .0;
    _lambdat.zero();
  
    for (uint32_t p = 0; p < _n; ++p) 
      for (uint32_t q = 0; q < _n; ++q)  {

	if (p >= q)
	  continue;

	process(p,q);
      }
    
    // mut
    for (uint32_t k = 0; k < _k; ++k)
      _mu[k] = _mu0 + SQ(_sigma0) * _mut[k];
    
    printf("mu %s\n", _mu.s().c_str());
    fflush(stdout);
    
    // lambda_n
    for (uint32_t n = 0; n < _n; ++n)  {
      _lambda[n] = _mu1 + SQ(_sigma1) * _lambdat[n];
      for (uint32_t k = 0; k < _k; ++k) 
	_gamma.set(n, k, _alpha[k] + _gammat.at(n,k));
    }
    set_dir_exp(_gamma, _Elogpi);
    _iter++;
    
    printf("iteration %d\n", _iter);
    estimate_pi();
    printf("logl=%f\n", approx_log_likelihood());
    //save_groups();
    save_model();
    compute_and_log_groups();
    heldout_likelihood();
  }
}

void
GLMNetwork::link_infer()
{
  assign_training_links();
  printf("nlinks = %d\n", _nlinks);
  fflush(stdout);
  
  set_dir_exp(_gamma, _Elogpi);
  _gammat.zero();
  _mut.zero();
  _sigma_betat = .0;
  _sigma_thetat = .0;
  _lambdat.zero();
  
  const double **linksd = _links.const_data();
  while (1) {
    
    for (uint32_t n = 0; n < _nlinks; ++n) {
      
      uint32_t p = linksd[n][0];
      uint32_t q = linksd[n][1];
      
      process(p,q);
      if (n % 100 == 0) {
	printf("\rprocessed %d links", n);
	fflush(stdout);
      }
    }
    printf("\n");
    
    // mut
    for (uint32_t k = 0; k < _k; ++k)
      _mu[k] = _mu0 + SQ(_sigma0) * _mut[k];
    
    printf("mu %s\n", _mu.s().c_str());
    fflush(stdout);
    
    // lambda_n
    for (uint32_t n = 0; n < _n; ++n)  {
      _lambda[n] = _mu1 + SQ(_sigma1) * _lambdat[n];
      for (uint32_t k = 0; k < _k; ++k) 
	_gamma.set(n, k, _alpha[k] + _gammat.at(n,k));
    }
    set_dir_exp(_gamma, _Elogpi);
    _iter++;
    
    if (_iter % _env.reportfreq == 0) {
      printf("iteration %d\n", _iter);
      estimate_pi();
      //printf("logl=%f\n", approx_log_likelihood());
      //save_groups();
      save_model();
      compute_and_log_groups();
      heldout_likelihood();
    }
  }
}

void
GLMNetwork::estimate_pi()
{
  const double ** const gd = _gamma.const_data();
  double **epid = _pi.data();
  for (uint32_t n = 0; n < _n; ++n) {
    double s = .0;
    for (uint32_t k = 0; k < _k; ++k)
      s += gd[n][k];
    assert(s);
    for (uint32_t k = 0; k < _k; ++k)
      epid[n][k] = gd[n][k] / s;
  }
}

void
GLMNetwork::save_groups()
{
  FILE *groupsf = fopen(Env::file_str("/groups.txt").c_str(), "w");
  char buf[32];
  const IDMap &seq2id = _network.seq2id();

  ostringstream sa;
  Array groups(_n);
  Array pi_i(_k);
  for (uint32_t i = 0; i < _n; ++i) {
    sa << i << "\t";
    IDMap::const_iterator it = seq2id.find(i);
    uint32_t id = 0;
    if (it == seq2id.end()) { // single node
      id = i;
    } else
      id = (*it).second;

    sa << id << "\t";
    _pi.slice(0, i, pi_i);
    double max = .0;
    for (uint32_t j = 0; j < _k; ++j) {
      memset(buf, 0, 32);
      sprintf(buf,"%.3f", pi_i[j]);
      sa << buf << "\t";
      if (pi_i[j] > max) {
	max = pi_i[j];
	groups[i] = j;
      }
    }
    sa << groups[i] << "\n";
  }
  fprintf(groupsf, "%s", sa.str().c_str());
  fflush(groupsf);
  fclose(groupsf);
}

void
GLMNetwork::save_popularity()
{
  FILE *degf = fopen(Env::file_str("/deg.txt").c_str(), "w");
  for (uint32_t n = 0; n < _n; ++n) {
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(n);
    if (idt != m.end()) {
      fprintf(degf,"%d\t%d\t%d\t%.5f\n", 
	      n, (*idt).second, _network.deg(n), _lambda[n]);
    }
  }
  fclose(degf);
}


void
GLMNetwork::save_mu()
{
  FILE *f = fopen(Env::file_str("/mu.txt").c_str(), "w");
  fprintf(f, "%d\t%.5f\n", 65536, _globalmu);
  for (uint32_t k = 0; k < _k; ++k)
    fprintf(f,"%d\t%.5f\n", k, _mu[k]);
  fclose(f);
}


void
GLMNetwork::save_model()
{
  FILE *gammaf = fopen(Env::file_str("/gamma.txt").c_str(), "w");
  FILE *hnodef = fopen(Env::file_str("/heldout-nodes.txt").c_str(), "w");
  const double ** const gd = _gamma.const_data();
  for (uint32_t i = 0; i < _n; ++i) {
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) {

      if (_heldout_deg[i] >= _network.deg(i)) {
	fprintf(hnodef, "%d\t%d\t%d\n", i, idt->second, _heldout_deg[i]);
	//printf("* Warning: node %d(%d) not used in training\n", 
	//(*idt).second, i);
	//continue;
      }

      fprintf(gammaf,"%d\t", i);
      debug("looking up i %d\n", i);
      fprintf(gammaf,"%d\t", (*idt).second);
      for (uint32_t k = 0; k < _k; ++k) {
	if (k == _k - 1)
	  fprintf(gammaf,"%.5f\n", gd[i][k]);
	else
	  fprintf(gammaf,"%.5f\t", gd[i][k]);
      }
    } else {
      printf("strange error!\n");
      fflush(stdout);
      assert(0);
    }
  }
  fclose(gammaf);
  fclose(hnodef);
  save_popularity();
  save_mu();
}


double
GLMNetwork::heldout_likelihood(bool nostop)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;

  for (SampleMap::const_iterator i = _heldout_map.begin();
       i != _heldout_map.end(); ++i) {
    
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

    yval_t y = _network.y(p,q);
    double u = pair_likelihood2(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    info("edge likelihood for (%d,%d) is %f\n", p,q,u);
  }
  double nshol = (_zeros_prob * (szeros / kzeros)) + (_ones_prob * (sones / kones));
  fprintf(_hf, "%d\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%.9f\t%.9f\n",
	  _iter, duration(), s / k, k,
	  szeros / kzeros, kzeros, sones / kones, kones,
	  _zeros_prob * (szeros / kzeros),
	  _ones_prob * (sones / kones),
	  nshol);
  fflush(_hf);

  // Use hol @ network sparsity as stopping criteria
  double a = nshol;
  bool stop = false;
  int why = -1;
  if (_iter > 1000) {
    if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
      stop = true;
      why = 100;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (a > _max_h)
      _max_h = a;
    
    if (_nh > 2) { // be robust to small fluctuations in predictive likelihood
      why = 1;
      stop = true;
    }
  }
  _prev_h = nshol;
  FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
  fprintf(f, "%d\t%d\t%.5f\t%.5f\t%d\n", 
	  _iter, duration(), 
	  a, _max_h, why);
  fclose(f);
  if (_env.use_validation_stop && stop) {
    do_on_stop();
    exit(0);
  }
  return a;
}

double
GLMNetwork::precision_likelihood(bool nostop)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  _degstats.clear();
  _ndegstats.clear();
  _vmap.clear();
  _nvmap.clear();

  FILE *df = fopen(Env::file_str("/degstats.txt").c_str(), "w");
  assert (df);

  for (SampleMap::const_iterator i = _precision_map.begin();
       i != _precision_map.end(); ++i) {
    
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

    yval_t y = _network.y(p,q);
    double u = pair_likelihood2(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    info("edge likelihood for (%d,%d) is %f\n", p,q,u);

    uint32_t pdeg = _network.deg(p);
    uint32_t qdeg = _network.deg(q);

    _degstats[(uint32_t)(pdeg / 10)] += u;
    _ndegstats[(uint32_t)(pdeg / 10)]++;

    _degstats[(uint32_t)(qdeg / 10)] += u;
    _ndegstats[(uint32_t)(qdeg / 10)]++;

    Edge dd(pdeg, qdeg);
    Network::order_edge(_env, dd);

    _vmap[dd] += u;
    _nvmap[dd]++;
  }
  double nshol = (_zeros_prob * (szeros / kzeros)) + (_ones_prob * (sones / kones));
  fprintf(_hf, "%d\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%.9f\t%.9f\n",
	  _iter, duration(), s / k, k,
	  szeros / kzeros, kzeros, sones / kones, kones,
	  _zeros_prob * (szeros / kzeros),
	  _ones_prob * (sones / kones),
	  nshol);
  fflush(_hf);
  
  for (DoubleMap::const_iterator i = _degstats.begin(); i != _degstats.end(); ++i) {
    uint32_t deg = i->first;
    double v = i->second;
    uint32_t n = _ndegstats[deg];
    fprintf(df, "%d\t%d\t%f\n", deg * 10, n, v / n);
  }
  fflush(df);
  fclose(df);
  
  FILE *vf = fopen(Env::file_str("/degpairstats.txt").c_str(), "w");
  assert (df);
  for (ValueMap::const_iterator i = _vmap.begin(); i != _vmap.end(); ++i) {
    const Edge &ee = i->first;
    uint32_t ep = ee.first;
    uint32_t eq = ee.second;
    double v = i->second;
    fprintf(vf, "%d\t%d\t%f\t%d\n", ep, eq, v / _nvmap[ee], _nvmap[ee]);
  }
  fclose(vf);

  // Use hol @ network sparsity as stopping criteria
  double a = nshol;
  return nshol;
}

double
GLMNetwork::validation_likelihood()
{
  if (_env.accuracy)
    return .0;

  return .0; // XXX

  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _validation_map.begin();
       i != _validation_map.end(); ++i) {

    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = get_y(p,q);
    bool seen = false; // TODO: fix heldout for sparse network
#endif

    assert (!seen);
    double u = pair_likelihood2(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    debug("edge likelihood for (%d,%d) is %f\n", p,q,u);
  }
  double nshol = (_zeros_prob * (szeros / kzeros)) + (_ones_prob * (sones / kones));
  fprintf(_vf, "%d\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%.9f\t%.9f\n",
	  _iter, duration(), s / k, k,
	  szeros / kzeros, kzeros, sones / kones, kones,
	  _zeros_prob * (szeros / kzeros),
	  _ones_prob * (sones / kones),
	  nshol);
  fflush(_vf);

  return s / k;
}

double
GLMNetwork::training_likelihood()
{
  if (_env.accuracy || !_env.log_training_likelihood)
    return .0;
  
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  uint32_t c = 0;

  for (SampleMap::const_iterator i = _training_map.begin();
       i != _training_map.end(); ++i) {
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);

#ifndef SPARSE_NETWORK
    const yval_t ** const yd = _network.y().const_data();
    yval_t y = yd[p][q] & 0x01;
    bool seen = yd[p][q] & 0x80;
#else
    yval_t y = _network.y(p,q);
    bool seen = false;
#endif

    if (!seen)
      c++;
    double u = pair_likelihood2(p,q,y);
    s += u;
    k += 1;
    if (y) {
      sones += u;
      kones++;
    } else {
      szeros += u;
      kzeros++;
    }
    debug("pair likelihood for (%d,%d) is %f\n", p,q,u);
  }
  fprintf(_trf, "%d\t%d\t%.5f\t%d\t%.5f\t%d\t%.5f\t%d\t%d\n",
	  _iter, duration(), s / k, k,
	  szeros / kzeros, kzeros, sones / kones, kones,c);
  fflush(_trf);
  return s / k;
}

double
GLMNetwork::approx_log_likelihood()
{
  set_dir_exp(_gamma, _Elogpi);
  const double * const alphad = _alpha.const_data();
  const double ** const elogpid = _Elogpi.const_data();
  const double ** const gd = _gamma.const_data();

  double s = .0, s1 = .0;
  double v = .0;
  for (uint32_t p = 0; p < _n; ++p) {
    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += gsl_sf_lngamma(alphad[k]);
    s += gsl_sf_lngamma(_alpha.sum()) - v;
    
    v = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      v += (alphad[k] - 1) * elogpid[p][k];
    }
    s += v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += gsl_sf_lngamma(gd[p][k]);
    s -= gsl_sf_lngamma(_gamma.sum(p)) - v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += (gd[p][k] - 1) * elogpid[p][k];
    s -= v;

    // theta
    s += - log(_sigma1) - (SQ(_lambda[p])+SQ(_sigma_theta)) / (2 * SQ(_sigma1));
    s += log(_sigma_theta);
  }
  s1 = s;
  debug("1:s1=%f\n", s1);

  double s2  = s;
  for (uint32_t k = 0; k < _k; ++k) {
    // beta
    s += - log(_sigma0) - (SQ(_mu[k])+SQ(_sigma_theta)) / (2 * SQ(_sigma0));
    s += - SQ(_mu0) / (2 * SQ(_sigma0)) + SQ(_mu0) / SQ(_sigma0);
    
    s -= - log(_sigma_beta) - (SQ(_mu[k]) + SQ(_sigma_beta))/(2 * SQ(_sigma_beta));
    s -= - SQ(_mu[k]) / (2*SQ(_sigma_beta)) + SQ(_mu[k])/ SQ(_sigma_beta);
  }
  s2 = s - s2;
  debug("2:s2=%f\n", s2);

  double s3 = s;
  for (uint32_t a = 0; a < _n; ++a) 
    for (uint32_t b = 0; b < _n; ++b) {

      if (a >= b)
	continue;

      yval_t y = _network.y(a,b);
      s += y * (_lambda[a] + _lambda[b]);

      _lc.reset(a,b,y);
      _lc.update_phi();

      const Matrix &phi = _lc.phi();
      const double ** const phid = phi.const_data();

      double u = .0, t = .0;
      for (uint32_t k = 0; k < _k; ++k) {
	s += y * _mu[k] * phid[k][k];
	u += phid[k][k] * exp(_mu[k] + SQ(_sigma_beta)/2);
	t += phid[k][k];
      }
      s += y * (1 - t) * _epsilon;
      debug("t = %f\n", t);
      debug("3: (%d:%d) s = %f\n", a, b, s);

      double w = exp(_lambda[a] + SQ(_sigma_theta)/2 + _lambda[b] + SQ(_sigma_theta)/2);
      w = w * (u + (1 - t) * exp(_epsilon));
      debug("w = %f\n", w);
      s -= log(1 + w);
      
      for (uint32_t k1 = 0; k1 < _k; ++k1)
	for (uint32_t k2 = 0; k2 < _k; ++k2) {
	  s += phid[k1][k2] * (elogpid[a][k1] + elogpid[b][k2]);
	  if (phid[k1][k2] > .0)
	    s -= phid[k1][k2] * log(phid[k1][k2]);
	}
      debug("4: y=%d (%d:%d) s = %f\n", y, a, b, s);
    }
  s3 = s - s3;
  debug("s1=%.3f,s2=%.3f,s3=%f,s=%.3f\n", s1, s2, s3, s);

  fprintf(_lf, "%d\t%d\t%.5f\n", _iter, duration(), s);
  fflush(_lf);
  return s;
}

void
GLMNetwork::estimate_pi(uint32_t p, Array &pi_p) const
{
  const double ** const gd = _gamma.data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += gd[p][k];
  assert(s);
  for (uint32_t k = 0; k < _k; ++k)
    pi_p[k] = gd[p][k] / s;
}

double
GLMNetwork::pair_likelihood(uint32_t p, uint32_t q, yval_t y) const
{
  Array pi_p(_k);
  Array pi_q(_k);
  
  estimate_pi(p, pi_p);
  estimate_pi(q, pi_q);

  debug("beta = %s\n", _beta.s().c_str());
  debug("lambda[%d] = %f\n", p, _lambda[p]);
  debug("lambda[%d] = %f\n", q, _lambda[q]);
  
  double s = .0;
  double u = .0;
  for (uint32_t k = 0; k < _k; ++k) {
    u += (pi_p[k] * pi_q[k] * _mu[k]); 
    s += pi_p[k] * pi_q[k];
  }
  u += _lambda[p] + _lambda[q];
  u += (1 - s) * _epsilon;
  debug("u = %f\n", u);
  double r = (double)1.0 / (1 + exp(-u));
  double z = gsl_ran_bernoulli_pdf(y, r);
  if (z < 1e-30)
    z = 1e-30;
  return log(z);
}

double
GLMNetwork::pair_likelihood2(uint32_t p, uint32_t q, yval_t y) const
{
  Array pi_p(_k);
  Array pi_q(_k);
  
  estimate_pi(p, pi_p);
  estimate_pi(q, pi_q);

  debug("beta = %s\n", _beta.s().c_str());
  debug("lambda[%d] = %f\n", p, _lambda[p]);
  debug("lambda[%d] = %f\n", q, _lambda[q]);

  double s = .0;
  double u = .0, z = .0, r = .0, m = .0;
  for (uint32_t k = 0; k < _k; ++k)  {
#ifdef GLOBAL_MU
    u = _lambda[p] + _lambda[q] + _globalmu;
#else
    u = _lambda[p] + _lambda[q] + _mu[k];
#endif
    r = (double)1.0 / (1 + exp(-u));
    z = gsl_ran_bernoulli_pdf(y, r);
    s += z * pi_p[k] * pi_q[k];
    m += pi_p[k] * pi_q[k];
  }
  u = _lambda[p] + _lambda[q] + _epsilon;
  r = (double)1.0 / (1 + exp(-u));
  z = gsl_ran_bernoulli_pdf(y, r);
  s += z * (1 - m);
  if (s < 1e-30)
    s = 1e-30;
  return log(s);
}

void
GLMNetwork::get_random_edge(bool link, Edge &e) const
{
  if (!link) {
    e.first = 0;
    e.second = 0;

    do {
      e.first = gsl_rng_uniform_int(_r, _n);
      e.second = gsl_rng_uniform_int(_r, _n);
      Network::order_edge(_env, e);
      assert(e.first == e.second || Network::check_edge_order(e));
    } while (!edge_ok(e));
  } else {
    do {
      const EdgeList &edges = _network.edges();
      int m  = gsl_rng_uniform_int(_r, _network.ones());
      e = edges[m];
      assert(Network::check_edge_order(e));
    } while (!edge_ok(e));
  }
}

void
GLMNetwork::compute_and_log_groups()
{
  Matrix fmap(_n,_k);
  double **fmapd = fmap.data();

  _communities.clear();
  uint32_t unlikely = 0;
  uint32_t c = 0;
  Array pi_i(_k);
  FILE *f = fopen(Env::file_str("/network.gml").c_str(), "w");
  fprintf(f, "graph\n[\n\tdirected 0\n");
  fflush(f);

  for (uint32_t i = 0; i < _n; ++i) {
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator idt = m.find(i);

    assert (idt != m.end());
    fprintf(f, "\tnode\n\t[\n");
    fprintf(f, "\t\tid %d\n", i);
    fprintf(f, "\t\textid %d\n", idt->second);
    fprintf(f, "\t\tpopularity %d\n", (int)(100 * _lambda[i]));
    fprintf(f, "\t\tgroup %d\n", most_likely_group(i));
    fprintf(f, "\t]\n");
    fflush(f);
  }

  for (uint32_t i = 0; i < _n; ++i) {
    _pi.slice(0, i, pi_i);
    
    Array pi_m(_k);
    const vector<uint32_t> *edges = _network.get_edges(i);

    for (uint32_t e = 0; e < edges->size(); ++e) {
      uint32_t m = (*edges)[e];
      if (i < m) {
	yval_t y = get_y(i,m);
	assert  (y == 1);
	c++;
	
	_pi.slice(0, m, pi_m);
	uint32_t max_k = 65535;
	double max = find_max_k(i, m, pi_i, pi_m, max_k);

	
	fmapd[i][max_k]++;
	fmapd[m][max_k]++;

	if (max > _env.link_thresh) {
	  if (fmapd[i][max_k] > _env.lt_min_deg)
	    _communities[max_k].push_back(i);
	  if (fmapd[m][max_k] > _env.lt_min_deg)
	    _communities[max_k].push_back(m);
	}
	
	fprintf(f, "\tedge\n\t[\n");
	fprintf(f, "\t\tsource %d\n", i);
	fprintf(f, "\t\ttarget %d\n", m);
	fprintf(f, "\t\tcolor %d\n", max_k);
	fprintf(f, "\t]\n");
	fflush(f);
	c++;
      }
    }
  }
  printf("unlikely = %d\n", unlikely);
  fflush(stdout);
  printf("c = %d\n", c);
  fflush(stdout);
  write_communities(_communities, "/communities.txt");
  fclose(f);

  if (_env.nmi) 
    compute_mutual("/communities.txt");
}

void
GLMNetwork::compute_mutual(string s)
{
  char cmd[1024];
  sprintf(cmd, "/usr/local/bin/mutual %s %s >> %s",
	  Env::file_str("/ground_truth.txt").c_str(),
	  Env::file_str(s.c_str()).c_str(),
	  Env::file_str("/mutual.txt").c_str());
  if (system(cmd) < 0)
    lerr("error spawning cmd %s:%s", cmd, strerror(errno));
}

double
GLMNetwork::find_max_k(uint32_t i, uint32_t j, 
		       Array &pi_i, Array &pi_j, uint32_t &max_k)
{
  double max = .0;
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k) {
    double logodds = _mu[k] + _lambda[i] + _lambda[j];
    debug("mu=%.3f, lambda i=%3f, lambda j=%3f, pi i=%.3f, pi j=%.3f\n", 
	  _mu[k], _lambda[i], _lambda[j], pi_i[k], pi_j[k]);
    // apply logit-inverse function
    double l = (1.0 / (1 + exp(-logodds))) * pi_i[k] * pi_j[k];
    s += l;
    if (l > max) {
      max = l;
      max_k = k;
    }
  }
  return max / s;
}

void
GLMNetwork::auc()
{
  FILE *f = fopen(Env::file_str("/auc.txt").c_str(), "w");
  for (SampleMap::const_iterator i = _precision_map.begin();
       i != _precision_map.end(); ++i) {
    
    const Edge &e = i->first;
    const IDMap &m = _network.seq2id();
    IDMap::const_iterator a = m.find(e.first);
    IDMap::const_iterator b = m.find(e.second);

    uint32_t p = a->second;
    uint32_t q = b->second;
    assert (p != q);
    
    yval_t y = _network.y(e.first,e.second);
    double a1 = 0, a2 = 0;
    double l1 = 0, l2 = 0;
    double u = link_prob(e.first, e.second, a1, a2, l1, l2);

    uint32_t pdeg = _network.deg(e.first);
    uint32_t qdeg = _network.deg(e.second);

    fprintf(f, "%d %.3f %d %d %d %d %.3f %.3f %.3f %.3f\n", 
	    y, u, p, q, pdeg, qdeg, a1, a2, l1, l2);
  }
  fclose(f);
  /*
  char cmd[1024];
  sprintf(cmd, "/usr/local/bin/roc < %s >> %s",
	  Env::file_str("/auc.txt").c_str(),
	  Env::file_str("/auc-summary.txt").c_str());
  if (system(cmd) < 0)
    lerr("error spawning cmd %s:%s", cmd, strerror(errno));
  sprintf(cmd, "/usr/local/bin/roc -plot < %s >> %s",
	  Env::file_str("/auc.txt").c_str(),
	  Env::file_str("/roc-curve.txt").c_str());
  if (system(cmd) < 0)
    lerr("error spawning cmd %s:%s", cmd, strerror(errno));
  sprintf(cmd, "/usr/local/bin/roc -prec < %s >> %s",
	  Env::file_str("/auc.txt").c_str(),
	  Env::file_str("/precision-recall-curve.txt").c_str());
  if (system(cmd) < 0)
    lerr("error spawning cmd %s:%s", cmd, strerror(errno));
  */

  FILE *g = fopen(Env::file_str("/auc2.txt").c_str(), "w");
  for (SampleMap::const_iterator i = _heldout_map.begin();
       i != _heldout_map.end(); ++i) {
    
    const Edge &e = i->first;
    uint32_t p = e.first;
    uint32_t q = e.second;
    assert (p != q);
    
    yval_t y = _network.y(p,q);
    double a1=0, a2=0;
    double l1 = 0, l2 = 0;
    double u = link_prob(p,q,a1,a2,l1,l2);

    fprintf(g, "%d %.3f\n", y, u);
  }
  fclose(g);

}

double
GLMNetwork::link_prob(uint32_t p, uint32_t q, 
		      double &u, double &r,
		      double &l1, double &l2) const
{
  Array pi_p(_k);
  Array pi_q(_k);
  
  estimate_pi(p, pi_p);
  estimate_pi(q, pi_q);

  debug("beta = %s\n", _beta.s().c_str());
  debug("lambda[%d] = %f\n", p, _lambda[p]);
  debug("lambda[%d] = %f\n", q, _lambda[q]);
  
  double s = .0;
  double z = .0, m = .0;
  u = .0; 
  r = .0;
  l1 = _lambda[p];
  l2 = _lambda[q];

  for (uint32_t k = 0; k < _k; ++k)  {
#ifdef GLOBAL_MU
    u = _lambda[p] + _lambda[q]  + _globalmu;
#else
    u = _lambda[p] + _lambda[q]  + _mu[k];
#endif
    r = (double)1.0 / (1 + exp(-u));
    z = gsl_ran_bernoulli_pdf(1.0, r);
    s += z * pi_p[k] * pi_q[k];
    m += pi_p[k] * pi_q[k];
  }
  u = _lambda[p] + _lambda[q]  + _epsilon;
  r = (double)1.0 / (1 + exp(-u));
  z = gsl_ran_bernoulli_pdf(1.0, r);
  s += z * (1 - m);
  return gsl_ran_bernoulli_pdf(1.0, s);
}
		       

void
GLMNetwork::write_communities(MapVec &communities, string name)
{
  const IDMap &seq2id = _network.seq2id();
  FILE *commf = fopen(Env::file_str(name.c_str()).c_str(), "w");
  //FILE *sizef = fopen(Env::file_str("/communities_size.txt").c_str(), "a");
  map<uint32_t, uint32_t> count;
  for (std::map<uint32_t, vector<uint32_t> >::const_iterator i = communities.begin();
       i != communities.end(); ++i) {
    //fprintf(commf, "%d\t", i->first);
    //uint32_t commid = i->first;
    const vector<uint32_t> &u = i->second;

    map<uint32_t, bool> uniq;
    vector<uint32_t> ids;
    vector<uint32_t> seq_ids;
    for (uint32_t p = 0; p < u.size(); ++p) {
      map<uint32_t, bool>::const_iterator ut = uniq.find(u[p]);
      if (ut == uniq.end()) {
	IDMap::const_iterator it = seq2id.find(u[p]);
	uint32_t id = 0;
	assert (it != seq2id.end());
	id = (*it).second;
	//fprintf(commf, "%d ", id);
	ids.push_back(id);
	seq_ids.push_back(u[p]);
	uniq[u[p]] = true;
      }
    }
    //gml(commid, seq_ids);

    uArray vids(ids.size());
    for (uint32_t j = 0; j < ids.size(); ++j)
      vids[j] = ids[j];
    vids.sort();
    for (uint32_t j = 0; j < vids.size(); ++j)
      fprintf(commf, "%d ", vids[j]);

    fprintf(commf, "\n");
    //fprintf(sizef, "%d\t%ld\n", i->first, uniq.size());
  }
  fclose(commf);
}

