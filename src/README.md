# Learning Gait
## Experiment 64
- forward walking experiment at maximum speed
- no goals
- no her
Run the following command to run:
```bash
python3 ddpg.py --experiment 64 --out_path rl/out_dir/models --env AntEnv --env_version 0
```
## Experiment 65
- forward walking experiment at variable speed
- 4 dimensional goal
- goal consisted of x-velocity, y-velocity, height, yaw
- her used
Run the following command to run:
```bash
python3 ddpg.py --experiment 65 --out_path rl/out_dir/models --env AntEnv --env_version 1 --her
```

## Experiment 66
- omnidirectional walking experiment at fixed speed
- 3 dimensional goal
- goal consisits of x position, y position, height
- no her
Run the following command to run:
```bash
python3 ddpg.py --experiment 66 --out_path rl/out_dir/models --env AntEnv --env_version 2
```

## Experiment 67
- omnidirectional walking experiment at multiple speeds and turning
- 6 dimensional goal
- goal consisits of x position, y position, x velocity, y velocity, height, yaw
- using her
Run the following command to run:
```bash
python3 ddpg.py --experiment 67 --out_path rl/out_dir/models --env AntEnv --env_version 3 --her
```

## Experiment 68
- omnidirectional walking at constant speed and yaw velocity
- 20 dimensional dict goal
- goal consists of 'command', 'ctrl', 'position', 'orientation'
- command consists of x velocity, y velocity, z velcoity, roll velocity, pitch velocity, yaw velocity
- ctrl consists of joint angle positions
- position consists of x coordinate, y coordinate, z coordinate
- orientation consists of x angle, y angle, z angle
- Action is tuple
- using her
- oscillator model
Run the following command to run:
```bash
python3 ddpg.py --experiment 68 --out_path rl/out_dir/models --env AntEnv --env_version 4 --her
```

## Experiment 69
- omnidirectional walking at constant speed and yaw velocity
- 20 dimensional dict goal
- goal consists of 'command', 'ctrl', 'position', 'orientation'
- command consists of x velocity, y velocity, z velcoity, roll velocity, pitch velocity, yaw velocity
- ctrl consists of joint angle positions
- position consists of x coordinate, y coordinate, z coordinate
- orientation consists of x angle, y angle, z angle
- Action is 8 dimensional
- using her
- fully connected model; no oscillators
Run the following command to run:
```bash
python3 ddpg.py --experiment 69 --out_path rl/out_dir/models --env AntEnv --env_version 5 --her
```
