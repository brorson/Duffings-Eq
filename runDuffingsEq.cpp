/* -----------------------------------------------------------------------------
 * This example solves Duffing's equation using Sundials CVODE.
 * The ODE system is given by
 *
 * y'' + delta*y' + alpha*y + beta*y^3 = gamma*cos(omega*t)
 *
 * Broken into two 1st order equations, the system to solve is
 *
 * dy1/dt = y2;
 * dy2/dt = -delta*y2 - alpha*y1 - beta*y1^2 + gamma*cos(omega*t);
 *
 * with initial condition y1 = -1 and y2 = -1 at t = 0.
 *
 * The main program calls the solver sub-fun using different values
 * of drive parameter gamma.  For small gamma the solution is periodic,
 * but the solution goes chaotic with increasing gamma.
 *
 * The program is compiled using g++, but most of the code looks like
 * straight C -- I cloned this example from the CVODE examples which
 * are written in C.  I use g++ since the plotting lib requires it.
 *
 * SDB -- Dec 2021.
 * ---------------------------------------------------------------------------*/

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <cvode/cvode.h>               /* access to CVODE                 */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */

// Stuff for plotting
#include <cmath>
#include <vector>
#include <sciplot/sciplot.hpp>
using namespace sciplot;


/* Precision specific formatting macros */
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"


/* Constants */
#define PI    RCONST(3.141592653589793238)
#define ZERO  RCONST(0.0)
#define ONE   RCONST(1.0)
#define MONE  RCONST(-1.0)

/* simulation parameters */
#define TSTOP  200.0   // Simulation run time
#define DELTAT 0.01    // Simulation time step
#define TREC   170.0   // Time to start recording points.
#define NUMPTS ceil((TSTOP-TREC)/DELTAT)

/* Duffing's eq parameters */
#define DELTA 0.3
#define ALPHA -1.0
#define BETA 1.0
#define OMEGA 1.2

/* User-defined data structure */
typedef struct UserData_
{
  realtype alpha;   /* Natural freq of undriven, linear osc */
  realtype beta;    /* Coefficient of nonlinearity */
  realtype delta;   /* Loss coefficient */
  realtype gamma;   /* Drive amplitude */
  realtype omega;   /* Drive freq */
  
  realtype rtol;    /* integration tolerances */
  realtype atol;

  realtype tstop;   /* time when to stop */
  realtype dt;      /* interval between recording points */

} *UserData;

/* Subfcn which integrates the ODE */
static int integrate_ode(SUNContext sunctx, UserData user_data,
			 std::vector<double> &tp, std::vector<double> &yp);

/* Functions provided to CVODE */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


/* Utility functions */
static int PrintUserData(UserData udata);
static int WriteOutput(realtype t, N_Vector y, realtype ec,
                       int screenfile, FILE *YFID);
static int PrintStats(void *cvode_mem);
static int check_retval(void *returnvalue, const char *funcname, int opt);


/* -----------------------------------------------------------------------------
 * Main Program -- this runs the loop over increasing values of gamma
 * ---------------------------------------------------------------------------*/
int main()
{
  SUNContext      sunctx     = NULL; /* SUNDIALS context     */
  UserData        udata      = NULL; /* user data structure        */  
  int             retval;
  
  std::vector<double> yp(NUMPTS);  // Used for plotting
  std::vector<double> tp(NUMPTS);  // Used for plotting

  /* Create the SUNDIALS context */
  // Since I run on a serial machine, the context is NULL.
  retval = SUNContext_Create(NULL, &sunctx);
  if(check_retval(&retval, "SUNContext_Create", 1)) return(1);

  /* Allocate and initialize user data structure */
  udata = (UserData) malloc(sizeof *udata);
  if (check_retval((void *)udata, "malloc", 0)) return(1);

  udata->delta = DELTA;
  udata->alpha = ALPHA;
  udata->beta = BETA;
  udata->omega = OMEGA;

  udata->rtol = RCONST(1.0e-4);
  udata->atol = RCONST(1.0e-9);

  udata->tstop = TSTOP;  /* Sim run time */
  udata->dt = DELTAT;      /* Data point period */

  // Create a Plot object and set up plotting
  Plot plot;
  
  // Set the legend
  plot.legend().hide();

  // Set the t and y labels
  plot.xlabel("t");
  plot.ylabel("y");
  
  // Iterate over gamma values to simulate.
  for (realtype gam = 0.2; gam<=0.34; gam+=0.02) {
    udata->gamma = gam;
  
    // Do the actual integration in this sub-fcn.
    integrate_ode(sunctx, udata, tp, yp);

    // Do plot of computed solution
    plot.drawCurve(tp, yp)
    .label("test")
    .lineWidth(1);

    std::cout << "gam = " << gam << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  // Show the plot in a pop-up window
  plot.size(1000, 400);
  plot.show();

  // Free memory before exiting.
  SUNContext_Free(&sunctx);

  return(0);
}


/* -----------------------------------------------------------------------------
 * Sub function -- this integrates the ODE and returns the time vector
 * tp and the solution vector yp for plotting.
 * ---------------------------------------------------------------------------*/
static int integrate_ode(SUNContext sunctx, UserData user_data,
			 std::vector<double> &tp, std::vector<double> &yp) {
  void            *cvode_mem = NULL; /* CVODE memory         */
  N_Vector         y         = NULL; /* solution vector      */
  realtype        *ydata     = NULL; /* solution vector data */
  SUNMatrix        A         = NULL; /* Jacobian matrix      */
  SUNLinearSolver  LS        = NULL; /* linear solver        */

  int      retval;          /* reusable return flag       */
  int      out      = 0;    /* output counter             */
  int      totalout = 0;    /* output counter             */
  int      idx;             /* count iterations of time stepping */
  realtype tout;
  realtype ec       = ZERO; /* constraint error           */
  realtype t;
  
  //FILE *YFID = NULL; /* solution output file */  

  /* Create serial vector to store the solution */
  y = N_VNew_Serial(2, sunctx);
  if (check_retval((void *)y, "N_VNew_Serial", 0)) return(1);

  /* Set initial contion */
  ydata    = N_VGetArrayPointer(y);
  ydata[0] = MONE;
  ydata[1] = MONE;

  /* Create CVODE memory.  Use BDF (backward differentiation) method */
  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (check_retval((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Initialize CVODE */
  /* CVodeInit(void *cvode_mem, CVRhsFn f, realtype t0, N_Vector y0) */
  retval = CVodeInit(cvode_mem, f, tout, y);
  if (check_retval(&retval, "CVodeInit", 1)) return(1);

  /* Attach user-defined data structure to CVODE */
  retval = CVodeSetUserData(cvode_mem, user_data);
  if(check_retval(&retval, "CVodeSetUserData", 1)) return(1);

  /* Set integration tolerances */
  retval = CVodeSStolerances(cvode_mem, user_data->rtol, user_data->atol);
  if (check_retval(&retval, "CVodeSStolerances", 1)) return(1);

  /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(2, 2, sunctx);
  if(check_retval((void *)A, "SUNDenseMatrix", 0)) return(1);

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(y, A, sunctx);
  if(check_retval((void *)LS, "SUNLinSol_Dense", 0)) return(1);

  /* Attach the matrix and linear solver to CVODE */
  retval = CVodeSetLinearSolver(cvode_mem, LS, A);
  if(check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

  /* Set a user-supplied Jacobian function */
  retval = CVodeSetJacFn(cvode_mem, Jac);
  if(check_retval(&retval, "CVodeSetJacFn", 1)) return(1);

  /* Set max steps between outputs */
  retval = CVodeSetMaxNumSteps(cvode_mem, 100000);
  if (check_retval(&retval, "CVodeSetMaxNumSteps", 1)) return(1);

  /* Output problem setup */
  retval = PrintUserData(user_data);
  if(check_retval(&retval, "PrintUserData", 1)) return(1);

  /* Output initial condition */
  //YFID = fopen("DuffingsEqSimulation.dat","w");
  //printf("\n     t            x              y          err constr\n");
  //WriteOutput(tout, y, ec, 1, YFID);
  
  /* Start time-stepping iteration */
  tout = user_data->dt;  // Target stop point of time step
  t = tout;          // Actual stop point of time step
  idx = 0;
  while (t <= user_data->tstop)
  {
    retval = CVodeSetStopTime(cvode_mem, tout);
    if (check_retval(&retval, "CVodeSetStopTime", 1)) return(1);

    /* Advance to new time tout */
    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    if (check_retval(&retval, "CVode", 1)) break;

    //WriteOutput(tout, y, ec, 1, YFID);
    
    // Add this point to plot vectors if simulation has gone
    // far enough.
    if (t >= TREC) {
      ydata = N_VGetArrayPointer(y);
      yp[idx] = static_cast<double>(ydata[0]);
      tp[idx] = static_cast<double>(t);
      // std::cout << "idx = " << idx << ", " << "tp = " << tp[idx] << ", " << "yp = " << yp[idx] << std::endl;
      idx++;
    }
    
    /* Update output time */
    tout += user_data->dt;
  }

  /* Close output file */
  //fclose(YFID);

  /* Print final solution to screen */
  //std::cout << "-------  Final output ------" << std::endl;
  //WriteOutput(tout, y, ec, 1, NULL);
  //if (check_retval(&retval, "WriteOutput", 1)) return(1);

  /* Print some final statistics about run */
  PrintStats(cvode_mem);

  /* Free memory */
  N_VDestroy(y);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  CVodeFree(&cvode_mem);

  return(0);
}

/* -----------------------------------------------------------------------------
 * Functions provided to CVODE
 * ---------------------------------------------------------------------------*/

//----------------------------------------------------------------
/* Compute the right-hand side function.  The calling args are in the 
 * form required by CVODE.  The ODE system is
 * dy1/dt = y2;
 * dy2/dt = -delta*y2 - alpha*y1 - beta*y1^2 + gamma*cos(omega*t);
 */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  // Extract calling args
  UserData  udata = (UserData) user_data;
  realtype *ydata = N_VGetArrayPointer(y);
  realtype *fdata = N_VGetArrayPointer(ydot);

  // Convenience variables
  realtype a = udata->alpha;
  realtype b = udata->beta;
  realtype d = udata->delta;
  realtype g = udata->gamma;
  realtype w = udata->omega;

  realtype y1 = ydata[0];
  realtype y2 = ydata[1];

  // ODE system evaluation
  fdata[0] = y2;
  fdata[1] = -d*y2 - a*y1 - b*y1*y1*y1 + g*cos(w*t);
  
  return(0);
}


//----------------------------------------------------------------
/* Compute the Jacobian of the right-hand side function, J(t,y) = df/dy */
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  UserData  udata = (UserData) user_data;
  realtype *ydata = N_VGetArrayPointer(y);
  realtype *Jdata = SUNDenseMatrix_Data(J);

  // Convenience variables
  realtype a = udata->alpha;
  realtype b = udata->beta;
  realtype d = udata->delta;
  realtype g = udata->gamma;
  realtype w = udata->omega;

  realtype y1 = ydata[0];
  realtype y2 = ydata[1];

  Jdata[0] = ZERO;
  Jdata[1] = ONE;
  Jdata[2] = -a - 2*b*y1*y1;
  Jdata[3] = -d;

  return(0);
}


/* -----------------------------------------------------------------------------
 * Private helper functions
 * ---------------------------------------------------------------------------*/


//------------------------------------------------------------------------
static int PrintUserData(UserData udata)
{
  if (udata == NULL) return(-1);

  // Convenience variables
  realtype a = udata->alpha;
  realtype b = udata->beta;
  realtype d = udata->delta;
  realtype g = udata->gamma;
  realtype w = udata->omega;

  
  printf("\nDuffing's equation example\n");
  printf("---------------------------------------------\n");
  printf("alpha      = %0.4" ESYM"\n", a);
  printf("beta       = %0.4" ESYM"\n", b);
  printf("delta      = %0.4" ESYM"\n", d);
  printf("gamma      = %0.4" ESYM"\n", g);
  printf("omega      = %0.4" ESYM"\n", w);  
  printf("---------------------------------------------\n");
  printf("rtol       = %" GSYM"\n", udata->rtol);
  printf("atol       = %" GSYM"\n", udata->atol);
  printf("tstop      = %f\n", udata->tstop);
  printf("---------------------------------------------\n");

  return(0);
}


//------------------------------------------------------------------------
/* Output the solution to the screen or disk */
static int WriteOutput(realtype t, N_Vector y, realtype ec,
                       int screenfile, FILE* YFID)
{
  realtype *ydata = N_VGetArrayPointer(y);

  if (screenfile == 1)
  {
    /* output solution to screen */
    printf("%0.4" ESYM" %14.6" ESYM" %14.6" ESYM" %14.6" ESYM"\n",
           t, ydata[0], ydata[1], ec);
  }
  else
  {
    /* check file pointers */
    if (YFID == NULL) return(1);

    /* output solution to disk */
    fprintf(YFID, "%24.16" ESYM " %24.16" ESYM " %24.16" ESYM "\n",
            t, ydata[0], ydata[1]);

  }

  return(0);
}


//------------------------------------------------------------------------
/* Print final statistics */
static int PrintStats(void *cvode_mem)
{
  int retval;
  long int nst, nfe, nsetups, nje, nni, ncfn, netf;

  retval = CVodeGetNumSteps(cvode_mem, &nst);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
  retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_retval(&retval, "CVodeGetNumErrTestFails", 1);
  retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

  retval = CVodeGetNumJacEvals(cvode_mem, &nje);
  check_retval(&retval, "CVodeGetNumJacEvals", 1);

  printf("\nIntegration Statistics:\n");

  printf("Number of steps taken = %-6ld\n", nst);
  printf("Number of function evaluations = %-6ld\n", nfe);

  printf("Number of linear solver setups = %-6ld\n", nsetups);
  printf("Number of Jacobian evaluations = %-6ld\n", nje);

  printf("Number of nonlinear solver iterations = %-6ld\n", nni);
  printf("Number of convergence failures = %-6ld\n", ncfn);
  printf("Number of error test failures = %-6ld\n", netf);

  return(0);
}

//------------------------------------------------------------------------
/* Check function return value */
static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Opt 0: Check if a NULL pointer was returned - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nERROR: %s() returned a NULL pointer\n\n",
            funcname);
    return(1);
  }
  /* Opt 1: Check if retval < 0 */
  else if (opt == 1)
  {
    retval = (int *) returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nERROR: %s() returned = %d\n\n",
              funcname, *retval);
      return(1);
    }
  }

  return(0);
}
