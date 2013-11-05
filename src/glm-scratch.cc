
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
