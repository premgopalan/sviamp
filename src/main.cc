#include "env.hh"
#include "glm.hh"
#include "log.hh"
#include <stdlib.h>

#include <string>
#include <iostream>
#include <sstream>
#include <signal.h>

string Env::prefix = "";
Logger::Level Env::level = Logger::DEBUG;
FILE *Env::_plogf = NULL;
void usage();
void test();

Env *env_global = NULL;

volatile sig_atomic_t sig_handler_active = 0;

void
term_handler(int sig)
{
  if (env_global) {
    printf("Got signal. Saving model and groups.\n");
    fflush(stdout);
    env_global->terminate = 1;
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

int 
main(int argc, char **argv)
{
  
  signal(SIGTERM, term_handler);
  
  bool run_gap = false;
  bool force_overwrite_dir = true;
  string datfname = "network.dat";
  string datdir = "";
  bool gen = false, ppc = false, lcstats = false;
  bool gml = false;
  string label = "";
  uint32_t nthreads = 0;
  int i = 1;
  uint32_t n = 0, k = 0;
  double ngscale = .0;
  double link_thresh = 0.5;
  bool nolambda = false;
  uint32_t nmemberships = 5;

  bool stratified = false, rnode = false, rpair = false;
  bool batch = false;
  bool online = true;
  bool nodelay = true;
  bool load = false; 
  bool adamic_adar = false; 
  uint32_t iter = 0;
  uint32_t lambda_iter = 0;
  uint32_t scale = 1;
  bool disjoint = false;
  bool orig = false;
  bool massive = false;
  bool single = false;
  bool sparsek = false;
  uint32_t lt_min_deg = 0;
  bool lowconf = false;

  int itype = 0; // init type
  string eta_type = "default"; // "default", "sparse", "regular" or "dense"
  uint32_t rfreq  = 100;
  bool accuracy = false;
  double stopthresh = 0.00001;

  double infthresh = 0;
  bool nonuniform = false;

  string ground_truth_fname = ""; // ground-truth communities file, if any
  bool nmi = false;
  bool bmark = false;
  bool randzeros = false;
  bool preprocess = false;
  bool strid = false;
  string groups_file = "";
  
  bool init_comm = false;
  string init_comm_fname = "";
  bool acc = false, lcacc = false;
  bool ammopt = false;
  bool onesonly = false;
  bool node_scaling_on = false;
  bool lpmode = false;
  bool gtrim  = false;
  bool fastinit = false;
  uint32_t max_iterations = 0;

  bool glm = false;
  bool glm2 = false;
  bool pcp = false;
  bool onlypop = false;
  bool postprocess = false;

  double hol_ratio = 0.01;
  double rand_seed = 0;

  if (argc == 1) {
    usage();
    exit(-1);
  }
  
  while (i <= argc - 1) {
    if (strcmp(argv[i], "-help") == 0) {
      usage();
      exit(0);
    } else if (strcmp(argv[i], "-gp") == 0) {
      fprintf(stdout, "+ gamma poisson model\n");
      run_gap = true;
    } else if (strcmp(argv[i], "-force") == 0) {
      fprintf(stdout, "+ overwrite option set\n");
      force_overwrite_dir = true;
    } else if (strcmp(argv[i], "-online") == 0) {
      fprintf(stdout, "+ online option set\n");
      online = true;
      batch = false;
    } else if (strcmp(argv[i], "-dir") == 0) {
      if (i + 1 > argc - 1) {
	fprintf(stderr, "+ insufficient arguments!\n");
	exit(-1);
      }
      datdir = string(argv[++i]);
      fprintf(stdout, "+ using file %s\n", datdir.c_str());
    } else if (strcmp(argv[i], "-ppc") == 0) {
      ppc = true;
      fprintf(stdout, "+ ppc option\n");
    } else if (strcmp(argv[i], "-lcstats") == 0) {
      lcstats = true;
      fprintf(stdout, "+ lc stats option\n");
    } else if (strcmp(argv[i], "-gml") == 0) {
      gml = true;
      fprintf(stdout, "+ gml option\n");
    } else if (strcmp(argv[i], "-gen") == 0) {
      gen = true;
      fprintf(stdout, "+ generate dataset option\n");
    } else if (strcmp(argv[i], "-stratified") == 0) {
      stratified = true;
      fprintf(stdout, "+ stratified option set\n");
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch = true;
      online = false;
      fprintf(stdout, "+ batch option set\n");
    } else if (strcmp(argv[i], "-nodelay") == 0) {
      nodelay = true;
      fprintf(stdout, "+ nodelay option set\n");
    } else if (strcmp(argv[i], "-rnode") == 0) {
      rnode = true;
      fprintf(stdout, "+ random node option set\n");
    } else if (strcmp(argv[i], "-rpair") == 0) {
      rpair = true;
      fprintf(stdout, "+ random pair option set\n");
    } else if (strcmp(argv[i], "-load") == 0) {
      load = true;
      iter = atoi(argv[++i]);
      lambda_iter = atoi(argv[++i]);
      fprintf(stdout, "+ load model option set "
	      "(iter = %d, lambda_iter = %d)\n", 
	      iter, lambda_iter);
    } else if (strcmp(argv[i], "-adamic-adar") == 0) {
      adamic_adar = true;
      fprintf(stdout, "+ adamic adar option set\n");
    } else if (strcmp(argv[i], "-scale") == 0) {
      scale = atoi(argv[++i]);
      fprintf(stdout, "+ scale set to %d\n", scale);
    } else if (strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
      fprintf(stdout, "+ n = %d\n", n);
    } else if (strcmp(argv[i], "-k") == 0) {
      k = atoi(argv[++i]);
      fprintf(stdout, "+ K = %d\n", k);
    } else if (strcmp(argv[i], "-disjoint") == 0) {
      disjoint = true;
    } else if (strcmp(argv[i], "-label") == 0) {
      label = string(argv[++i]);
    } else if (strcmp(argv[i], "-nthreads") == 0) {
      nthreads = atoi(argv[++i]);
      fprintf(stdout, "+ nthreads = %d\n", nthreads);
    } else if (strcmp(argv[i], "-orig") == 0) {
      orig = true;
    } else if (strcmp(argv[i], "-massive") == 0) {
      massive = true;
      fprintf(stdout, "+ massive option set\n");
    } else if (strcmp(argv[i], "-single") == 0) {
      single = true;
      fprintf(stdout, "+ stochastic blockmodel option set\n");
    } else if (strcmp(argv[i], "-sparsek") == 0) {
      sparsek = true;
      fprintf(stdout, "+ fastamm sparsek option set\n");
    } else if (strcmp(argv[i], "-itype") == 0) {
      itype = atoi(argv[++i]);
      fprintf(stdout, "+ itype = %d\n", itype);
    } else if (strcmp(argv[i], "-eta-type") == 0) {
      eta_type = string(argv[++i]);
      fprintf(stdout, "+ eta-type = %s\n", eta_type.c_str());
    } else if (strcmp(argv[i], "-nmi") == 0) {
      ground_truth_fname = string(argv[++i]);
      fprintf(stdout, "+ ground truth fname = %s\n",
	      ground_truth_fname.c_str());
      nmi = true;
    } else if (strcmp(argv[i], "-rfreq") == 0) {
      rfreq = atoi(argv[++i]);
      fprintf(stdout, "+ rfreq = %d\n", rfreq);
    } else if (strcmp(argv[i], "-accuracy") == 0) {
      accuracy = true;
      fprintf(stdout, "+ accuracy = %d\n", accuracy);
    } else if (strcmp(argv[i], "-stopthresh") == 0) {
      stopthresh = atof(argv[++i]);
      fprintf(stdout, "+ stopthresh = %.6f\n", stopthresh);
    } else if (strcmp(argv[i], "-inf") == 0) {
      infthresh = atof(argv[++i]);
      fprintf(stdout, "+ infthresh = %.6f\n", infthresh);
    } else if (strcmp(argv[i], "-nonuniform") == 0) {
      nonuniform = true;
      fprintf(stdout, "+ nonuniform sampling mode\n");
    } else if (strcmp(argv[i], "-bmark") == 0) {
      bmark = true;
      fprintf(stdout, "+ benchmark mode\n");
    } else if (strcmp(argv[i], "-randzeros") == 0) {
      randzeros = true;
      fprintf(stdout, "+ randzeros mode\n");
    } else if (strcmp(argv[i], "-preprocess") == 0) {
      preprocess = true;
      massive = true;
      fprintf(stdout, "+ preprocess network option \n");
    } else if (strcmp(argv[i], "-acc") == 0) {
      acc = true;
    } else if (strcmp(argv[i], "-lcacc") == 0) {
      lcacc = true;
    } else if (strcmp(argv[i], "-strid") == 0) {
      strid = true;
      fprintf(stdout, "+ strid mode\n");
    } else if (strcmp(argv[i], "-groups-file") == 0) {
      groups_file = string(argv[++i]);
      fprintf(stdout, "+ groups_file = %s\n", groups_file.c_str());
    } else if (strcmp(argv[i], "-glm") == 0) {
      fprintf(stdout, "+ glm option set\n");
      glm = true;
    } else if (strcmp(argv[i], "-pcp") == 0) {
      fprintf(stdout, "+ pcp option set\n");
      pcp = true;
    } else if (strcmp(argv[i], "-onlypop") == 0) {
      fprintf(stdout, "+ onlypop option set\n");
      onlypop = true;
    } else if (strcmp(argv[i], "-glm2") == 0) {
      fprintf(stdout, "+ glm2 option set\n");
      glm2 = true;
    } else if (strcmp(argv[i], "-postprocess") == 0) {
      fprintf(stdout, "+ postprocess option set\n");
      postprocess = true;
    } else if (strcmp(argv[i], "-ngscale") == 0) {
      ngscale = atof(argv[++i]);
      fprintf(stdout, "+ ngscale = %.3f\n", ngscale);
    } else if (strcmp(argv[i], "-link-thresh") == 0) {
      link_thresh = atof(argv[++i]);
      fprintf(stdout, "+ link_thresh = %.3f\n", link_thresh);
    } else if (strcmp(argv[i], "-lt-min-deg") == 0) {
      lt_min_deg = atof(argv[++i]);
      fprintf(stdout, "+ lt_min_deg = %d\n", lt_min_deg);
    } else if (strcmp(argv[i], "-lowconf") == 0) {
      lowconf = true;
      fprintf(stdout, "+ lowconf = %d\n", lowconf);
    } else if (strcmp(argv[i], "-nolambda") == 0) {
      nolambda = true;
      fprintf(stdout, "+ nolambda = %d\n", nolambda);
    } else if (strcmp(argv[i], "-v") == 0) {
      nmemberships = atoi(argv[++i]);
      fprintf(stdout, "+ nmemberships = %d\n", nmemberships);
    } else if (strcmp(argv[i], "-amm") == 0) {
      ammopt = true;
      batch = false;
      fprintf(stdout, "+ amm opt set\n");
    } else if (strcmp(argv[i], "-onesonly") == 0) {
      onesonly = true;
      fprintf(stdout, "+ ones only set\n");
    } else if (strcmp(argv[i], "-I") == 0) {
      init_comm = true;
      init_comm_fname = string(argv[++i]);
      fprintf(stdout, "+ loading initial communities from %s\n", init_comm_fname.c_str());
    } else if (strcmp(argv[i], "-node-scaling-on") == 0) {
      node_scaling_on = true;
      fprintf(stdout, "+ node scaling off\n");
    } else if (strcmp(argv[i], "-lpm") == 0) {
      lpmode = true;
      fprintf(stdout, "+ lp mode on\n");
    } else if (strcmp(argv[i], "-gtrim") == 0) {
      gtrim = true;
      fprintf(stdout, "+ gtrim node on\n");
    } else if (strcmp(argv[i], "-fastinit") == 0) {
      fastinit = true;
      fprintf(stdout, "+ fastinit mode\n");
    } else if (strcmp(argv[i], "-max-iterations") == 0) {
      max_iterations = atoi(argv[++i]);
      fprintf(stdout, "+ max iterations %d\n", max_iterations);
    } else if (strcmp(argv[i], "-heldout-ratio") == 0) {
      hol_ratio = atof(argv[++i]);
      fprintf(stdout, "+ hol ratio %.3f\n", hol_ratio);
    } else if (strcmp(argv[i], "-seed") == 0) {
      rand_seed = atof(argv[++i]);
      fprintf(stdout, "+ random seed set to %.5f\n", rand_seed);
    } else {
      fprintf(stdout, "unknown option %s!", argv[i]);
      assert(0);
    } 
    ++i;
  };

  datfname = datdir + "/train.tsv";
  
  assert (!(batch && online));
  
  Env env(n, k, massive, 
	  hol_ratio, rand_seed,
	  single, batch, stratified, 
	  nodelay, rpair, rnode, load, adamic_adar,
	  scale, disjoint,
	  force_overwrite_dir, datdir, 
	  (ppc || gml), run_gap, gen, label, nthreads, itype, eta_type,
	  nmi, ground_truth_fname, rfreq, 
	  accuracy, stopthresh, infthresh, 
	  nonuniform, bmark, randzeros, preprocess, 
	  strid, groups_file, glm, glm2, pcp, 
	  postprocess,
	  acc, lcacc, ngscale, link_thresh,
	  lt_min_deg, lowconf, nolambda, nmemberships, ammopt, 
	  onesonly, init_comm, init_comm_fname, node_scaling_on,
	  lpmode, gtrim, fastinit, max_iterations);

  env_global = &env;
  Network network(env);
  if (network.read(datfname.c_str()) < 0) {
    fprintf(stderr, "error reading %s; quitting\n", datfname.c_str());
    return -1;
  }

  fprintf(stdout, "network: n = %d, ones = %d, singles = %d\n", 
	  network.n(), 
	  network.ones(), network.singles());

  env.n = network.n() - network.singles();

  if (glm) {
    GLMNetwork glmnetwork(env, network);
    glmnetwork.infer();
    exit(0);
  }
}

void
usage()
{
  fprintf(stdout, "Stochastic AMM Tool\n"
	  "infer [OPTIONS]\n"
	  "\t-help\t\tusage\n"
	  "\t-file <name>\tread tab-separated edge file\n"
	  "\t-n <N>\t\tspecify number of nodes in network\n"
	  "\t-k <K>\t\tspecify number of communities\n"
	  "\t-batch\t\trun batch variational inference\n"
	  "\t-stratified\t\tuse stratified random pair sampling\n"
	  "\t-rnode\t\tuse random node sampling\n"
	  "\t-rpair\t\tuse random pair sampling\n"
	  "\t-label\t\tchoose a descriptive tag for output directory\n"
	  "\t-massive\t\tfor large datasets\n"
	  "\t-preprocess\t\tpreprocess large datasets\n"
	  "\t-rfreq\t\tset the frequency at which logging (of heldout-likelihood etc.) is done\n"
	  );
  fflush(stdout);
}
