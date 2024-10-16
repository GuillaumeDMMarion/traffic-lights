"""
Playable SUMO environment for human baseline comparison.
"""

from pynput import keyboard
from tlrl.sim.env import SumoEnv

print("")
print("Click on 'Run' to start...!")

env = SumoEnv(
    network="sumo/eval/0/intersection.net.xml",
    config="sumo-env.cfg",
    render=True,
    verbose=False,
    rnd_seed=58736,
)
env.reset()

print("")
print(
    "NOTE: increase simulation delay to 50, zoom in a bit and click back on this window!"
)
input("Press any key to continue...")
print("")
print(
    f"GOAL: let as few vehicle wait as little as possible over a {env.end_time}s time horizon. Vehicle wait time is visualized as a blue-red color gradient."
)
print("")
print("CONTROLS:")
print("w/s for South-West <> North-East.")
print("a/d for South-West > North-West, and North-East > South-East.")
print("up/down for North-West <> South-East.")
print("left/right for South-East > South-West, and North-West > North-East.")
print("ESC for premature simulation exit.")
print("")


def on_press(key):
    try:
        if hasattr(key, "char"):  # Check if key is a KeyCode object
            if key.char.lower() == "w":
                action = 0
            elif key.char.lower() == "s":
                action = 0
            elif key.char.lower() == "a":
                action = 1
            elif key.char.lower() == "d":
                action = 1
        elif key == keyboard.Key.up:
            action = 2
        elif key == keyboard.Key.down:
            action = 2
        elif key == keyboard.Key.left:
            action = 3
        elif key == keyboard.Key.right:
            action = 3
        else:
            return

        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        if done:
            print("total reward:", sum(env.reward))
            print("episode disutility:", info["disutility_sum"])
            print("vehicle disutility:", info["disutility_avg"])
            return False
    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
