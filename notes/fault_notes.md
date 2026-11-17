# Hydraulic test rig: fault notes

Illustrative notes written for this demo project. They are NOT manufacturer guidance;
a qualified technician must confirm any diagnosis before work is done.

## Cooler: reduced efficiency or near total failure
The cooler removes heat from the oil in the cooling-filtration circuit. When it degrades,
the temperature sensors TS1 to TS4 read higher, and the virtual sensors cooling efficiency
(CE) and cooling power (CP) fall. Reduced efficiency (state 20) is a warning; near total
failure (state 3) means the oil can overheat. Inspect the cooler fins and fan, the coolant
flow, and the oil temperature trend. Clean or replace the cooler if the temperature stays high.

## Valve: switching lag or close to total failure
The proportional valve controls the load pressure. A small lag (state 90) or severe lag
(state 80) shows up as slow pressure response during load changes in the pressure sensors
PS1 to PS6 and in the volume flow. Close to total failure (state 73) means the valve barely
switches. Check the pilot supply and the solenoid, look for oil contamination, and inspect or
replace the valve spool.

## Pump: internal leakage
Internal pump leakage lowers the delivered volume flow (FS1, FS2), causes pressure drops,
raises motor power (EPS1) and heats the oil, so temperature sensors can also rise. Weak
leakage (state 1) should be scheduled for inspection; severe leakage (state 2) reduces
performance quickly. Check the case drain flow and the pump seals, and plan a pump
overhaul or replacement for severe leakage.

## Accumulator: reduced pre-charge pressure
The hydraulic accumulator buffers pressure peaks. A lower gas pre-charge (115 bar slightly
reduced, 100 bar severely reduced, 90 bar close to total failure, 130 bar optimal) causes
larger pressure oscillations and changes in the pressure sensors PS1 to PS6 during load
changes. Measure the nitrogen pre-charge with a gauge, inspect the bladder or piston seal,
and re-charge or replace the accumulator.

## Sensor guide
PS1 to PS6 are pressure sensors (bar). EPS1 is motor power (W). FS1 and FS2 are volume flow
(l/min). TS1 to TS4 are temperatures (degrees C). VS1 is vibration (mm/s). CE is cooling
efficiency (%), CP is cooling power (kW) and SE is an efficiency factor (%); these three are
virtual sensors calculated from the measurements.

## Priority guidance
A component in a near-failure state, or more than one severe component, is high priority:
inspect within the current shift and consider stopping the rig. A single severe component is
high priority within 24 hours. Mild degradation is normal priority and can be scheduled
into the next planned maintenance window.

## Interaction between faults
Components affect each other. Pump leakage heats the oil and can look like a cooler fault;
a weak accumulator can look like a valve problem because both change pressure behaviour.
When several components are flagged, rely on the classifier confidence and inspect the
highest-confidence severe component first.

## Limits of this diagnosis
The anomaly score only says that a cycle deviates from normal behaviour. The component
classifier estimates which parts are degraded from one cycle of sensor statistics. Both can
be wrong, so every draft work order needs approval by a human before anything is filed.