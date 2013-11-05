
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



