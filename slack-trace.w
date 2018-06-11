// -*- c++ -*-

#define UNW_LOCAL_ONLY
#include <cstdio>
#include <cstring>             // memset
// #include <cstdint>             // uint64_t and friends
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <math.h>

#include <omp.h>

#include <libunwind.h>          // stack walking
#include <openssl/sha.h>        // sha hash

#include <timing.h> // used for adept timing

#include "slack-trace.h"

using namespace std;

#define INITIAL_STATIC_FRACTION 0.10

#define SEARCH_SF 1

#define SOLVE_SF 1

#define MATH_ENV_VAR_NAME "MATH_OPTION"

#define STATIC_FRACTION_ENV_VAR_NAME "STATIC_FRACTION"
#define TASKLET_SIZE_ENV_VAR_NAME "TASKLET_SIZE"

#define DEQUEUE_TIME_ENV_VAR_NAME "DEQUEUE_TIME"
#define NOISE_LENGTH_ENV_VAR_NAME "NOISE_LENGTH"

#define T_p_ENV_VAR_NAME "T_parallel"

#define SLACK_SCALE_ENV_VAR_NAME "SLACK_SCALE"
#define NOISE_SCALE_ENV_VAR_NAME "NOISE_SCALE"

#define SLACK_CONSCIOUS_ENV_VAR_NAME "SLACK_CONSCIOUS"
#define SLACK_THRESH_ENV_VAR_NAME    "SLACK_THRESH"

#define TRACE_SLACK_ENV_VAR_NAME     "TRACE_SLACK"
#define MAX_TRACE_SIZE_ENV_VAR_NAME  "MAX_TRACE"

#define CHECK_SLACK_ERROR_ENV_VAR_NAME  "TRACE_SLACK_ERROR"

#define CHECK_SLACK_OVHD_ENV_VAR_NAME  "TRACE_SLACK_OVHD"

#define DEFAULT_SLACK_THRESH 0.0001
#define NUM_WARMUP_ITERS 1000
#define STATIC_FRACTION_MIN 0.90
#define DEFAULT_DEQUEUE_OVERHEAD 0.00001


#define WARMUP_WORK 4096

//#define TRACE_OVHD 1

// TODO: INITIALIZE ME WITH A SENSIBLE VALUE!!!
// AS IN DIVIDE BY NUMBER OF ITERATIONS BASED ON MICROBENCHMARK.


double dequeue_overhead_per_iteration;

double measured_dequeue_overhead_per_iteration;

static double slackPredTime = 0.0;
static double static_fraction = -1.0;
static bool slack_conscious = false;
static bool check_slack_ovhd = false;
static bool check_slack_error = false;
static double slack_thresh = 0.00001;

static int math_option = 0;

static bool trace_slack = false;
static int  max_trace_size = -1;

static double noiseLength = 0.1;

static double T_parallel = 1.0 ;

static double noiseScale = 1.0;
static double slackScale = 1.0;

static int num_cores = 16;

static int taskletSize  = 1;
static double t_q = DEFAULT_DEQUEUE_OVERHEAD; // this includes dequeue overhead, and cost of coherence cache miss

static double dyn_ratio = 0.5;

double t_schedOvhd = 0.0;
double t_noise = 0.0;

static double static_frac_min = (1.0 - noiseLength/T_parallel);
static int rank;

int verbose = 0;

struct Trace {
  vector<double> times;
  double lastSlackTime;
  Trace(size_t size = 1000) : lastSlackTime(0.0) { }
  void add_time(double time) {
    if (trace_slack && (max_trace_size < 0 || times.size() < max_trace_size)) {
      times.push_back(time);
    }
    lastSlackTime = time;
  }
};

struct TraceOvhd {
  vector<double> timesOvhd;
  double lastSlackTimeOvhd;
  TraceOvhd(size_t size = 1000) : lastSlackTimeOvhd(0.0) { }
  void add_time(double time) {
    if (trace_slack && (max_trace_size < 0 || timesOvhd.size() < max_trace_size)) {
      timesOvhd.push_back(time);
    }
    lastSlackTimeOvhd = time;
  }
  void incr_time(double time) {
    if (trace_slack && (max_trace_size < 0 || timesOvhd.size() < max_trace_size)) {
      timesOvhd[timesOvhd.size() -1] +=  time;
      lastSlackTimeOvhd = timesOvhd[timesOvhd.size() -1];
    }
    else
      lastSlackTimeOvhd = time; //TODO:  need to figure out better solution here
  }
};

struct TraceErr {
  vector<double> timesErr;
  double lastSlackTimeErr;
  TraceErr(size_t size = 1000) : lastSlackTimeErr(0.0) { }
  void add_time(double time) {
    if (trace_slack && check_slack_error && (max_trace_size < 0 || timesErr.size() < max_trace_size)) {
      timesErr.push_back(time);
    }
    lastSlackTimeErr = time;
  }

  void incr_time(double time) {
    if (trace_slack && check_slack_error && (max_trace_size < 0 || timesErr.size() < max_trace_size)) {
      timesErr[timesErr.size() -1] +=  time;
      lastSlackTimeErr = timesErr[timesErr.size() -1];
    }
    else
      lastSlackTimeErr = time; //TODO:  could make this unified with slack time values structure
  }
};

typedef map<mpi_op_t, Trace> TraceMap; // the first entry here must change to the hash type.

typedef map<char*, Trace> TraceHashMap; // the first entry here must change to the hash type

typedef map<char*, TraceOvhd> TraceHashMapOvhd; // this is the overhead of slack prediction for each collective invocation
typedef map<char*, TraceErr> TraceHashMapErr; // this is the overhead of slack prediction for each collective invocation

static TraceMap mpi_coll_traces;
static TraceHashMap traces;
static TraceHashMap tracesSimple;

static TraceHashMapOvhd tracesOvhd;
static TraceHashMapOvhd tracesOvhd2;
static TraceHashMapOvhd tracesOvhd3;
static TraceHashMapErr tracesErr;

static string op_to_string(mpi_op_t op) {
  switch (op) {
    {{forallfn foo}}
  case {{foo}}_op: return string("{{foo}}"); {{endforallfn}}
  default:
    return string("unknown");
  }
}

static long unsigned int hash_to_string(struct hash h) {
  // uncomment this to check actual hash
  //cout << "first 8 bytes of hash: " << (long unsigned int)(h.hash) << endl 
  //TODO: need a hash to collID here
  long unsigned int myHashString = ((long unsigned int) (h.hash));
  return myHashString;
 }

mpi_op_t string_to_op(string opString) {
  {{forallfn foo}}
  if (opString == "{{foo}}" || opString == "{{foo}}_op") return {{foo}}_op;{{endforallfn}}
  return MPI_Invalid_op;
}

void write_out_traces_for_this_process() {
  int rank;
  int size;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PMPI_Comm_size(MPI_COMM_WORLD, &size);
  if(verbose)
    printf("writing out traces for rank %d \n", rank );

#ifdef TRACE_OVHD
  TraceHashMapOvhd::iterator i2 = tracesOvhd.begin();
  TraceHashMapOvhd::iterator i3 = tracesOvhd2.begin();
  TraceHashMapOvhd::iterator i4 = tracesOvhd3.begin();
#endif

#ifdef TRACE_ERR
  TraceHashMapErr::iterator i5 = tracesErr.begin();
#endif

  for (TraceHashMap::iterator i = traces.begin(); i != traces.end(); i++) {
    char* collId = i->first;
    Trace& trace = i->second;

    #ifdef TRACE_OVHD
    TraceOvhd& traceOvhd = i2->second;
    TraceOvhd& traceOvhd2 = i3->second;
    TraceOvhd& traceOvhd3 = i4->second;
    #endif

    #ifdef TRACE_ERR
    TraceErr& traceErr = i5->second ;
    #endif
    ostringstream filename;
    filename << "slackTrace"<< "." << ((long unsigned int) collId) << "." <<  size << "." << rank;
    ofstream file(filename.str().c_str());
    file << "#\t" << "iter"  <<  "\t"  << "slack" ;
#ifdef TRACE_OVHD
    file << "\t"  << "ovhd1" << "\t" <<  "ovhd2" << "\t" <<  "ovhd3" << "\t" << "totOvhd";
#endif

#ifdef TRACE_ERR
      file << "\t" << "err" ;
#endif
      file << endl;
      for (size_t x = 0; x < trace.times.size(); x++)
      {
        file  << "\t" << x << "\t"  << trace.times[x] << 

#ifdef TRACE_OVHD
          "\t" << traceOvhd.timesOvhd[x] <<
          "\t" << traceOvhd2.timesOvhd[x] <<
          "\t" << traceOvhd3.timesOvhd[x] <<
          "\t" << traceOvhd.timesOvhd[x] + traceOvhd2.timesOvhd[x]+ traceOvhd3.timesOvhd[x] <<
#endif
#ifdef TRACE_ERR
          "\t" << traceErr.timesErr[x] <<
#endif
          endl;
      }
  }
#ifdef TRACE_OVHD
  i2++;
  i3++;
  i4++;
  #endif

#ifdef TRACE_ERR
  i5++;
#endif
}


#define NUM_LOOP_TIMINGS 5


typedef struct
{
  int startIndex;
  int endIndex;
  int lastThread;   /* locality tag. This indicates which thread this tasklet ran on in the last iteration */
  int done;         /* indicates whether this piece of work is done or not */
  /* double lastIterThreadTime ;  */  /*  indicates the amount of time it took for this tasklet to complete in the previous iteration. With this information coupled with the knowle\
dge of the thread it last ran on as well as the time taken for a thread to complete its static portion, we can determine how much noise happened on a core within the execution of t\
he static work or the dynamic work.  */
} AppTasklet;

typedef struct WorkQueue
{
  int curr;
  int numTasklets;
  pthread_mutex_t workQueueMutex;
  AppTasklet** appWorkQueue;
};

struct LoopTimeRecord {
  double loop_times[NUM_LOOP_TIMINGS];
  size_t cur_entry;
  bool has_records;
  size_t dynamic_iterations;
  double schedOvhd;

  LoopTimeRecord() : cur_entry(0), has_records(false) {
    // initialize loop_timints so that get_min_loop_time works even
    // before we fill up the array.
    for (size_t i=0; i < NUM_LOOP_TIMINGS; i++) {
      loop_times[i] = numeric_limits<double>::max();
    }
  }

  /// start timing the current loop execution, store start time in next entry.
  void start() {
    loop_times[cur_entry] = get_time_ns();
  }

  // Stop timing the current loop execution and record the elapsed time.
  // The timing we get will include dynamic iterations, so we subtract out
  // dequeue overhead.  Caller must compute and pass in dequeue overhead.
  //
  // See endLoop() for that implementation.
  void end(double dequeue_overhead) {
    loop_times[cur_entry] = get_time_ns() - loop_times[cur_entry];
    loop_times[cur_entry] -= dequeue_overhead;
    cur_entry = (cur_entry + 1) % NUM_LOOP_TIMINGS;
    has_records = true;
    schedOvhd = dequeue_overhead;
  }

  // Returns the minimum of the last 5 recorded timings.
  // PRE: should not be called between start() and end()

  // switched from timing_t to double

  double min_loop_time() {
    return min(loop_times[0], loop_times[NUM_LOOP_TIMINGS]);
   }

 double get_dynamic_iters() {
   return dynamic_iterations;
   }
 };

long get_timing_adagioStyle()
{
  return get_time_ns();
}



double getMax(double* myNums, int n)
{
  double max = -1.0;
  if (n <= 0)
  {
    printf("Error: can't get maximum of an array of size zero or negative number \n");
    return -1.0;
  }
  int i;
  max = myNums[0];
  for (i = 0 ; i< n ; i++)
  {
    if(myNums[i] > max)
      max = myNums[i];
  }
  return max;
}

double getMin(double* myNums, int n)
{
  double min = 99999.0;
  if (n <= 0)
  {
    printf("Error: can't get minimum of an array of size zero or negative number \n");
    return -1.0;
  }
  int i;
  min = myNums[0];
  for (i = 0 ; i< n ; i++)
  {
    if(myNums[i] < min)
      min = myNums[i];
  }
  return min;
}

double getAvg(double* myNums, int n)
{
  double avg = 0.0;
  double sum = 0.0;
  int i;
  for (i = 0 ; i< n ; i++)
  {
    sum += myNums[i];
  }
  avg += sum/(n*1.0);
  return avg;
}


double guessStaticFractionAdagioStyle(LoopTimeRecord **loop_record_ptr)
 {
   if(verbose)
     printf("calling guessStaticFractionAdagioStyle!\n");
   // if this is the first time through, need to allocate a loop time record.
   if (!*loop_record_ptr) {
     *loop_record_ptr = new LoopTimeRecord();
   }

   if(verbose)
     printf("allocated looptime Record!  Static fraction is : %f\n", static_fraction);

   LoopTimeRecord& loop_record(**loop_record_ptr);  // get a reference for convenience
//   // if the loop has never been timed, start the first timing and
//   // return the initial static fraction
   if (!loop_record.has_records) {
     loop_record.start();
     if(verbose)
       printf("finished calling guessStaticFractionAdagioStyle! Static fraction is : %f\n", static_fraction);
     return INITIAL_STATIC_FRACTION;
   }

//   // if we get here then we have at least one good timing of the loop.
//   // so we can estimate the execution time using LoopTimeRecord::min_loop_time()

   double static_fraction_prime = 1.0;
   // slackTime = getSlackPredAdagioStyle(); //since we now know which collective we are referring to, we obtain the predicted slack
   // we put the code directly here to avoid function call overhead
   double slackTime = 0.0;
   double timeOvhd2 = -get_time_ns();

   struct hash h;
   memset(h.hash, 0, 20);
   get_stackwalk_hash(&h); //  this is where overhead can occur, so we need to possibly put some timers here.
   slackTime = traces[h.hash].lastSlackTime; // could possibly use minimum here. We can also do a risk analysis here and choose . Note
//   // that we use the entire hash to identify the collective. This hash consists of the instruction pointer and stack pointer, but it identifies
//   // a unique collective . Make sure we see the slack from a previous application timestep here.
   timeOvhd2 += get_time_ns();

   tracesOvhd2[h.hash].add_time(timeOvhd2);
//   /* printf("getSlackPred(): rank %d\t slackTime: %f \n", rank, slackTime); */
   int rank;
   int done = 0;
   double timeOvhd3 = -get_time_ns();
   tracesOvhd3[h.hash].add_time(timeOvhd3);
   PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (static_fraction < 0.0)
   {
    // compute automatically because we need to guess */
//     /* stats */
//     /* compute the static fraction */
     static_fraction = STATIC_FRACTION_MIN;
   }
  if (slack_conscious == true)
   {
     if (slack_thresh >= 0.00000 ) /*  if slack_thresh is not -2.0, we use the imitation strategy */
     {
       if( slackTime > slack_thresh)
         static_fraction = 1.0;
     }
     else if ( slack_thresh == -2.0 )  /* we use the proper strategy */
     {
 #ifdef SF_SEARCH
#endif

 #ifdef SOLVE_SF
//       // t parallel is estimated as the minimum execution time of the last 5 executions of this loop.
       double T_parallel =  loop_record.min_loop_time();
       double numItersLoop = (noiseScale*(1.0 - static_fraction))/loop_record.get_dynamic_iters();
       double dequeue_overhead_per_iteration = measured_dequeue_overhead_per_iteration;

#ifdef NEW_WAY
       static_fraction_prime  = 1.0 -  min(0.0, (1 - static_fraction) - ((slackScale*slackTime) / ((num_cores-1)*T_parallel - numItersLoop*dequeue_overhead_per_iteration )));
#endif

#endif
    }
    /* do proper slack risk analysis along with integration with app perf perf model later */
  }
  /*  check that static_fraction is a ratio */
  if(static_fraction > 1.0) // if it happens to be greater than 1.0 , we set it to 1.0
    static_fraction = 1.0;
  if(static_fraction < 0.0)
    static_fraction = static_frac_min;
  timeOvhd3 += get_time_ns();
  //timeOvhd3 +=  PMPI_Wtime();
  tracesOvhd3[h.hash].add_time(timeOvhd3);

 // start the timing for this execution of the loop, which should start immediately after the
  // call to guessStaticFractionAdagioStyle.
  loop_record.start();
  // return computed static fraction

  if(verbose)
    printf("finished calling guessStaticFractionAdagioStyle! Static fraction is : %f\n", static_fraction);

  return static_fraction;
}

// Instrumentation function that marks the end of a loop in client code.  Client needs to
// pass in a pointer to a static loop_record_ptr, along with the number of dynamic iterations
// that were executed by the loop.  We multiply this by the dequeue overhead per iteration
// obtained by the microbenchmark in MPI_Init() to get the actual dequeue overhead for the thing
// we just timed.
//
// It is erroneous to call this function without first calling guessStaticFractionAdagioStyle()
// before the loop.  Yes, that is a horrible naming convention and the functions do more than
// they should b/c we're trying to minimize the number of calls we make into the runtime library.
// We'll figure out something more intuitive later.
//
// TODO: rename this and guessStaticFractionAdagioStyle().
//
double endLoop(LoopTimeRecord **loop_record_ptr, int dynamic_iterations) {

  if(verbose)
    printf("calling endloop function!\n");
  if (!*loop_record_ptr) {
    cerr << "ERROR: endLoop called with uninitialized loop_record_ptr" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  LoopTimeRecord& loop_record(**loop_record_ptr);
  double dequeue_overhead = dequeue_overhead_per_iteration * dynamic_iterations;
  loop_record.end(dequeue_overhead);

  if(verbose)
    printf("finished calling endloop function! \t dynamic iterations are: %d \n", dynamic_iterations);
}

double loopEnd(LoopTimeRecord **loop_record_ptr, int dynamic_iterations) {
  return endLoop(loop_record_ptr, dynamic_iterations);
}

struct hash getNextCollective()
{
  struct hash h;
  //below is where we check the stacktrace and see which collective is coming up, and look up the stack trace to see the last invocation of it to get the predicted slack.
  // Note that this will require an invocation to hash_backtrace , which will return a hash for that MPI call. From that, we must use a hash function to actually obtain
  // the collective that is coming up (which we know, since it is stored on the stack, but is just not executed yet).
  // Then , we must transform that collective into the mpi_op_t type. ( Once we do that, our traces have the slack information, and the slack for this upcoming colective is predicted through that analysis of the traces -- this could involve a risk analysis, but we leave that out for now).
  memset(h.hash, 0, 20);
  get_stackwalk_hash(&h);
  return h;
}

/* usage:
   put the op that is coming up right after the openMP pragma.
   getSlackPred(MPI_Allreduce_op);
   You can also use this function within this library, when we are trying to guess the static fraction.
*/
double getSlackPred(mpi_op_t op)
{
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double slackTime = 0.0;
  /* slackTime = traces[op].values.front();  */
  slackTime = mpi_coll_traces[op].lastSlackTime;
  /* printf("getSlackPred(): rank %d\t slackTime: %f \n", rank, slackTime); */
  return slackTime;
}

double getSlackPredAdagioStyle()
{
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double slackTime = 0.0;

  struct hash h;
  memset(h.hash, 0, 20);
  /* slackTime = traces[op].values.front();  */
  get_stackwalk_hash(&h); //  this is where overhead can occur, so we need to possibly put some timers here.
  slackTime = traces[h.hash].lastSlackTime; // could possibly use minimum here. We can also do a risk analysis here and choose . Note
  // that we use the entire hash to identify the collective. This hash consists of the instruction pointer and stack pointer, but it identifies
  // a unique collective . Make sure we see the slack from a previous application timestep here.
  /* printf("getSlackPred(): rank %d\t slackTime: %f \n", rank, slackTime); */
  return slackTime;
}

//  This function obtains the hash value of the of the collective function invocation. It walks up the stack and gets the registers for the instruction pointer and stack pointer. 
//  THe result is stored in the 'hash' struct below.
void get_stackwalk_hash ( struct hash *h  ) {
       // Required for libunwind ( man -S3 libunwind )
       unw_cursor_t cursor; unw_context_t uc;
       unw_word_t regs[2];     // regs[0] holds ip, regs[1] holds sp;
       // Required for sha-1 hash ( man -S3 sha )
       SHA_CTX sc;
       // libunwind initialization
       unw_getcontext(&uc);
       unw_init_local(&cursor, &uc);
       // SHA-1 initialization
       SHA1_Init(&sc);
       // walk up the stack.
       while (unw_step(&cursor) > 0) {
               unw_get_reg(&cursor, UNW_REG_IP, &regs[0]);
               unw_get_reg(&cursor, UNW_REG_SP, &regs[1]);
               SHA1_Update( &sc, (const void *)regs, 2*sizeof(unw_word_t));
       }
       // move hash to our struct.
       SHA1_Final( (unsigned char *)h->hash, &sc);
       //printf("finished function get_stackwalk_hash \n");
}

//this is called in the application, and should be executed by one MPI process. The result is a file with all collectives stats in one file.

static double parse_double(const string& str) {
  char *err;
  double value = strtod(str.c_str(), &err);
  if (*err != '\0') {
    if (rank == 0) {
      cerr << str << " is not a valid floating point number" << endl;
    }
    PMPI_Abort(MPI_COMM_WORLD, 1);
  }
  return value;
}

static long parse_long(const string& str) {
  char *err;
  double value = strtol(str.c_str(), &err, 10);
  if (*err != '\0') {
    if (rank == 0) {
      cerr << str << " is not a valid integer" << endl;
    }
    PMPI_Abort(MPI_COMM_WORLD, 1);
  }
  return value;
}

static bool parse_bool(const string& str) {
  string data(str);
  std::transform(data.begin(), data.end(), data.begin(), ::tolower);
  if (data != "true" && data != "false") {
    if (rank == 0) {
      cerr << str << " is not a valid boolean value" << endl;
    }
    PMPI_Abort(MPI_COMM_WORLD, 1);
  }
  return (data == "true");
}

{{fn func MPI_Init}} {
  {{callfn}};
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Arch params
  int i;

  // ------- Dequeue time -------
  const char* env = getenv(DEQUEUE_TIME_ENV_VAR_NAME);
  if (env) {
    string dequeue_time_str(env);
    if (dequeue_time_str == "guess")
    {
     t_q = -1.0;
    }
    else
    {
      t_q = parse_double(dequeue_time_str);
    }
  }

  double* a = (double*) malloc(sizeof(double)*100*num_cores);
  double* b =  (double* ) malloc(sizeof(double)*100*num_cores);
  double* c = (double*) malloc(sizeof(double)*100*num_cores);

#pragma omp parallel for shared(a, b, c)
  for (i = 0 ; i< 100*num_cores; i++)
  {
    a[i] =  (i+1)*11.0;
    b[i] = i*3.0;
    c[i] = 7.0;
  }

  double staticTime = -get_time_ns();
#pragma omp parallel for
  for(i = 0 ; i < 100*num_cores ; i++ )
    c[i] = (a[i]/b[i] + a[i-1]/b[i-1] + a[i+1]/b[i+1])/c[i];
  staticTime += get_time_ns();
  double dynamicTime = - get_time_ns();
#pragma omp parallel for  schedule(dynamic)
  for(i = 0; i< 100*num_cores ; i++)
    c[i] = (a[i]/b[i] + a[i-1]/b[i-1] + a[i+1]/b[i+1])/c[i];
  dynamicTime += get_time_ns();


  double* comp_times = (double*) malloc(sizeof(double)*NUM_WARMUP_ITERS);
  int warmupIter;
  //omp_set_num_threads(omp_get_num_procs());
  for (warmupIter = 0; warmupIter < NUM_WARMUP_ITERS; warmupIter++)
  {
#pragma omp parallel shared(a, b, c, chunk) num_threads(16)
    {
#pragma omp for private(i) schedule(static)
      for(i = 0; i< 100*num_cores ; i++)
        c[i] = sqrt((a[i]/b[i] + a[i-1]/b[i-1] + a[i+1]/b[i+1])/c[i]);
    }
    comp_times[warmupIter] += MPI_Wtime();

  }
    T_parallel = getMin(comp_times, NUM_WARMUP_ITERS);
    noiseLength = getMax(comp_times, NUM_WARMUP_ITERS) - T_parallel;
    if(verbose)
      printf("measured noise length on rank %d is :  %f \n" ,rank , noiseLength);

    measured_dequeue_overhead_per_iteration =  (dynamicTime - staticTime)/(100*num_cores*1.0);
    if(verbose)
      printf("measured dequeue overhead per iteration on rank %d is : %f \n", rank, measured_dequeue_overhead_per_iteration);

  // ------- Noise length -------
  env = getenv(NOISE_LENGTH_ENV_VAR_NAME);
  if (env) {
    string noiseLength_str(env);
    if (noiseLength_str == "guess")
    {
      noiseLength = -1.0;
    }
    else
    {
      noiseLength = parse_double(noiseLength_str);
    }
  }

// -- MATH
  env = getenv(MATH_ENV_VAR_NAME);
  string math_str(env);
  if(math_str == "old" )
  {
    math_option  = 0;
  }
  else
    math_option = 1;

  // Runtime
  // ------- Static fraction -------
  env = getenv(STATIC_FRACTION_ENV_VAR_NAME);
  if (env) {
    string static_fraction_str(env);
    if (static_fraction_str == "guess")
    {
      static_fraction = -1.0;
    }
    else
    {
      static_fraction = parse_double(static_fraction_str);
    }
  }

  // ------- Tasklet size-------
  env = getenv(TASKLET_SIZE_ENV_VAR_NAME);
  if (env) {
    string tasklet_size_str(env);
    if (tasklet_size_str == "guess")
    {
      taskletSize = -1.0; //should be an integer
     }
    else
    {
      taskletSize = parse_long(tasklet_size_str);
    }
  }

  // ------- slack scale factor ----
  env = getenv(SLACK_SCALE_ENV_VAR_NAME);
  if (env) {
    string slackScale_str(env);
    slackScale = parse_double(slackScale_str);
  }

  // ------- noise scale factor ----
  env = getenv(NOISE_SCALE_ENV_VAR_NAME);
  if (env) {
    string noiseScale_str(env);
    noiseScale = parse_double(noiseScale_str);
  }

  // Application
  // ------- T_parallel -------
  env = getenv(T_p_ENV_VAR_NAME);
  if (env) {
    string T_p_str(env);
    if (T_p_str == "guess")
    {
      T_parallel = 1.0;
    }
    else
    {
      T_parallel = parse_double(T_p_str);
    }
  }

  // ------- Slack Conscious -------
  env = getenv(SLACK_CONSCIOUS_ENV_VAR_NAME);
  if (env) {
    slack_conscious = parse_bool(env);
  }

  // -------  Slack time -------------

#ifdef SLACK_NAIVE
  double slackTimeNaive = - get_time_ns();
  MPI_Barrier(MPI_COMM_WORLD);
  slackTimeNaive += get_time_ns();
  tracesSimple.add_time(slackTimeNaive);
#endif

  // ------- Slack prediction overhead -------
  env = getenv(CHECK_SLACK_OVHD_ENV_VAR_NAME);
  if (env) {
    string slack_ovhd(env);
    std::transform(slack_ovhd.begin(), slack_ovhd.end(), slack_ovhd.begin(), ::tolower);
    check_slack_ovhd = (slack_ovhd != "false");
  }

  // ------- Slack prediction error -------
  env = getenv(CHECK_SLACK_ERROR_ENV_VAR_NAME);
  if (env) {
    string slack_err(env);
    std::transform(slack_err.begin(), slack_err.end(), slack_err.begin(), ::tolower);
    check_slack_error = (slack_err != "false");
  }

  // ------- Slack Threshold -------
  env = getenv(SLACK_THRESH_ENV_VAR_NAME);
  if (env) {
    string slack_thresh_str(env);
    slack_thresh = -1.0;
    if (slack_conscious) {
      if (slack_thresh_str == "none") {
        slack_thresh = 0.0000;
      }
      else if (slack_thresh_str == "auto") {
        slack_thresh = -2.0;
      }
      else if (slack_thresh_str == "default") {
        slack_thresh = DEFAULT_SLACK_THRESH;
      }
      else {
        slack_thresh = parse_double(slack_thresh_str);
      }
    }
  }

  // ------- Toggle tracing on/off -------
  env = getenv(TRACE_SLACK_ENV_VAR_NAME);
  if (env) {
    trace_slack = parse_bool(env);
  }

  // ------- Max trace size -------
  env = getenv(MAX_TRACE_SIZE_ENV_VAR_NAME);
  if (env) {
    max_trace_size = parse_long(env);
  }

  if(rank == 0)
  {
    cout << "===========================================================" << endl;
    cout << "=== Running Vivek's slack conscious scheduling library. ===" << endl;
    cout << "===========================================================" << endl;
    cout << "SLACK_CONSCIOUS:  " << (slack_conscious ? "true" : "false") << endl;
    cout << "STATIC_FRACTION:  " << static_fraction << endl;
    cout << "NOISE_LENGTH:  " << noiseLength << endl;
    cout << "CHECK_SLACK_ERROR: " << check_slack_error << endl;
    cout << "CHECK_SLACK_OVHD: " << check_slack_ovhd << endl;
    cout << "SLACK_THRESH:     " << slack_thresh << endl;
    cout << "TRACE_SLACK:      " <<  (trace_slack ? "true" : "false") << endl;
    if (trace_slack) {
      cout << "MAX_TRACE_SIZE:   " <<  max_trace_size << endl;
    }
    cout << "===========================================================" << endl;
  }
  }{{endfn}}

/* note: we need to use the pred slacktimes everytime we call a particular function in the application */

{{fn func MPI_Finalize}} {
  if (trace_slack) {
    write_out_traces_for_this_process();
  }
  {{callfn}}
 } {{endfn}}

{{fn func MPI_Reduce MPI_Allreduce MPI_Scatter MPI_Gather MPI_Alltoall MPI_Allgather MPI_Bcast MPI_Barrier}} {

  struct hash h;
  memset(h.hash, 0, 20);
  get_stackwalk_hash(&h);

  double slackPrediction = 0.0;
  double actualSlackTime = 0.0;
  double timeErr = 0.0;

  double time = - get_time_ns();
  {{callfn}}
  time +=  get_time_ns();


  double timeOvhd = - get_time_ns();

  traces[h.hash].add_time(time);  // add slack time to table

  if(check_slack_ovhd)
    {
      timeOvhd += get_time_ns();
      tracesOvhd[h.hash].add_time(timeOvhd);
    }

  if(check_slack_error) {
    slackPrediction = traces[h.hash].lastSlackTime;
    actualSlackTime = time;
    if((slackPrediction - actualSlackTime) > 0 ) // if actual slack less than predicted, our scheduler should have been adjusted to have more dynamic scheduling, but we
      // didn't do that, so this is an error. Because the overprediction gives more room for excess work, we are less likely (*) to amplify. (TODO: reword this to clarify )
    {
      timeErr = (slackPrediction - time );
      tracesErr[h.hash].add_time(timeErr);
    }
  }
} {{endfn}}
