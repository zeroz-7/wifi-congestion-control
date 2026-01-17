#include "control.h"
#include "ns3/core-module.h"

using namespace ns3;

static Time g_controlInterval = Seconds (1.0);

void
ControlStep ()
{
  // Placeholder for now
  NS_LOG_UNCOND ("[CONTROL] Time = " << Simulator::Now ().GetSeconds ()
                                    << "s");

  // Schedule next control step
  Simulator::Schedule (g_controlInterval, &ControlStep);
}

void
StartControlLoop ()
{
  Simulator::Schedule (g_controlInterval, &ControlStep);
}
