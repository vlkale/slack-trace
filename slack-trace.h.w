// // -*- c -*-
#ifndef slack_histo_included_
#define slack_histo_included_

typedef enum {
 {{forallfn foo}}
 {{foo}}_op,{{endforallfn}}
  MPI_Invalid_op
 } mpi_op_t;

struct hash{   // could name this better
  char hash[20];
};

struct LoopTimeRecord;
struct WorkQueue;

#ifdef __cplusplus
extern "C" {
#endif

  double getMin(double*, int);
  double getMax(double*, int);
  double getAvg(double*, int);

// Thread scheduling functions

  void resetWorkQueue(struct WorkQueue *workQueue);
  void enqueueTasklets(struct WorkQueue *workQueue);

  double guessStaticFractionAdagioStyle(struct LoopTimeRecord **loop_record);

  double endLoop(struct LoopTimeRecord **loop_record, int dynamic_iterations);

  double loopEnd(struct LoopTimeRecord **loop_record, int dynamic_iterations);

  double guessStaticFraction(mpi_op_t op );

  long get_timing_adagioStyle();

  double getSlackPred(mpi_op_t op );
  double getSlackPredAdagioStyle();
  struct hash getNextCollective();
  mpi_op_t getDefaultCollective();
  void get_stackwalk_hash ( struct hash *h);

#ifdef __cplusplus
}
#endif

extern "C" double guessstaticfractionadagiostyle_(struct LoopTimeRecord** lr){return guessStaticFractionAdagioStyle(lr); }

extern "C" double loopend_(struct LoopTimeRecord** lr , int dyn_iters ){return loopEnd(lr, dyn_iters); }

#endif //slack_histo_included_
