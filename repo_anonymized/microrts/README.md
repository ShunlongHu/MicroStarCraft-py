# MicroRTS

## IEEE-CoG2023 MicroRTS competition

_Technical details in technical-description.md_

_Note: Anonymized for blind review. Links are broken, and code is possibly broken. Final
version will link to public repo._

### Agent installation instructions

1. Java (tested 11+) and Python 3.8+ must be installed
2. Download the AnonymizedAI archive. For the CoG2023 MicroRTS competition this can be
   downloaded from ANONYMIZED_URL
3. Unzip the archive: `unzip -j AnonymizedAI-0.0.38-bugfix.zip`
4. Upgrade and install Python depdendencies:

```
python -m pip install --upgrade pip
python -m pip install setuptools==65.5.0 wheel==0.38.4
python -m pip install --upgrade torch
```

5. Install the `.whl` file:

```
python -m pip install --upgrade WHL_FILE
```

The above steps makes `rai_microrts` callable within the terminal. `AnonymizedAI.java`
uses this to start a Python child process, which is used to compute actions.

### Win-Loss Against Prior Competitors on Public Maps

AnonymizedAI regularly beats prior competition winners and baselines on 7 of 8
competition public maps. The exception is the largest map (64x64). Each cell below represents the average result
of AnonymizedAI against the opponent AI for 100 matches (50 each as player 1 and player
2). A win is +1, loss is -1, and draw is 0. Same number of wins and losses would average to a
score of 0. A score of 0.9 corresponds to winning 95% of games (assuming no draws).

| map                     | POWorkerRush | POLightRush | CoacAI | Mayari | Map Total |
| :---------------------- | -----------: | ----------: | -----: | -----: | --------: |
| basesWorkers8x8A        |         0.91 |           1 |   0.98 |      1 |      0.97 |
| FourBasesWorkers8x8     |            1 |           1 |      1 |   0.97 |      0.99 |
| NoWhereToRun9x8         |            1 |           1 |   0.93 |   0.97 |      0.98 |
| basesWorkers16x16A      |            1 |           1 |   0.78 |   0.97 |      0.94 |
| TwoBasesBarracks16x16   |            1 |        0.78 |   0.98 |      1 |      0.94 |
| DoubleGame24x24         |            1 |           1 |   0.85 |      1 |      0.96 |
| BWDistantResources32x32 |            1 |        0.84 |   0.82 |   0.97 |      0.91 |
| (4)BloodBath.scmB       |         0.96 |          -1 |     -1 |     -1 |     -0.51 |
| AI Total                |         0.98 |         0.7 |   0.67 |   0.74 |      0.77 |

Mayari was the 2021 COG winner (prior competition), and CoacAI was the 2020 COG winner. POWorkerRush
and POLightRush are baseline AIs. POWorkerRush, POLightRush, and CoacAI use the default AStarPathFinding.

The round-robin tournamnet was run on a 2018 Mac Mini with Intel i7-8700B CPU (6-core,
3.2 GHz) with PyTorch limited to 6 threads. The avearge
execution time per turn varied by map-size with the shortest being NoWhereToRun9x8 (9
milliseconds) and longest BWDistantResources32x32 and BloodBath (22 ms). The tournament enforces 100 ms
per turn. AnonymizedAI exceeded the 100 ms limit and needed to skip its turn on less than
0.001% of turns but did lose by timeout in 5 matches (4 BloodBath [1% of BloodBath
matches], 1 BWDistantResources [0.25%])

### v0.0.38 vs v0.0.38-bugfix

The bugfix build only has a change in the Java-side AnonymizedAI where the thread pool is
allowed to empty to 0 threads. This fixes an issue where scripts would not terminate
because of threads awaiting tasks in AnonymizedAI. The bug does not affect gameplay at all.
The fix only changes the program end behavior. The Python `.whl` file is exactly the same
between v0.0.38 and v0.0.38-bugfix.

If you want to continue using v0.0.38
and want to fix the script termination behavior, add `System.exit(0);` at the end of the
`main` function.

### Best models variant

`AnonymizedAIBestModels.java` is a subclass of `AnonymizedAI`, which always uses the best
model for the given map. This still respects the `timeBudget` passed into it, so if the
model takes over the `timeBudget` it'll instead return an empty PlayerAction.

`AnonymizedAI` has 3 "general" models for 3 different sets of map sizes:

1. Maps whose longest dimension is 16
2. Maps whose longest dimension is 17-32
3. Maps whose longest dimension is 33-64

Additionally, there are models finetuned on specific maps:

1. maps/NoWhereToRun9x8.xml
2. maps/DoubleGame24x24.xml
3. maps/BWDistantResources32x32.xml (2 models: one larger, the other faster)

During the `preGameAnalysis` step, `AnonymizedAI` computes the next action to both warm
up data structures and to estimate how long each model is likely to take.
It will pick the largest model that will likely take less than 75% of the `timeBudget`,
which on slower machines will be the smaller, faster model. `AnonymizedAIBestModels` will
always pick the larger model.

If you want to run the agent with the best models, either make sure the machine is fast
enough to generally complete turns in under 50 milliseconds (Apple M1 Max, Intel Core
i7-8700B, Intel Xeon 8358 are all examples that are easily fast enough) OR increase the
timeBudget sufficiently high. `AnonymizedAIBestModels` shouldn't be necessary, but can be
used to ensure it's always used.
