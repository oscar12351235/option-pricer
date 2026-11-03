/ Smoke test for a local q/kdb+ installation.
/ Run from the repository root with:
/   q scripts/smoke.q

show "q smoke test";

result:sum 1 2 3;
expected:6;

show enlist[`result]!enlist result;

if[result<>expected;
  '"expected sum 1 2 3 to equal 6"
 ];

show "ok";
\\
