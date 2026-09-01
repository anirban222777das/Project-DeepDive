"""
Play Seaquest yourself with keyboard controls!

Controls:
  Arrow Keys  = Move (Up/Down/Left/Right)
  Space       = Fire
  Arrow + Space = Move and Fire simultaneously
  Q / ESC     = Quit
"""
import gymnasium as gym
import ale_py
import pygame
import sys
import time

gym.register_envs(ale_py)

# Seaquest action mapping:
# 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN
# 6=UPRIGHT, 7=UPLEFT, 8=DOWNRIGHT, 9=DOWNLEFT
# 10=UPFIRE, 11=RIGHTFIRE, 12=LEFTFIRE, 13=DOWNFIRE
# 14=UPRIGHTFIRE, 15=UPLEFTFIRE, 16=DOWNRIGHTFIRE, 17=DOWNLEFTFIRE

def get_action_from_keys(keys):
    up    = keys[pygame.K_UP]
    down  = keys[pygame.K_DOWN]
    left  = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    fire  = keys[pygame.K_SPACE]

    if up and right and fire: return 14
    if up and left and fire:  return 15
    if down and right and fire: return 16
    if down and left and fire:  return 17
    if up and fire:   return 10
    if right and fire: return 11
    if left and fire:  return 12
    if down and fire:  return 13
    if up and right:   return 6
    if up and left:    return 7
    if down and right: return 8
    if down and left:  return 9
    if fire:  return 1
    if up:    return 2
    if right: return 3
    if left:  return 4
    if down:  return 5
    return 0  # NOOP

def main():
    # Initialize pygame FIRST, before creating the gym environment
    pygame.init()
    
    # Create the environment with human rendering (ALE opens its own SDL window)
    env = gym.make("ALE/Seaquest-v5", render_mode="human")
    obs, _ = env.reset()

    print("=" * 40)
    print("  SEAQUEST — YOU ARE PLAYING!")
    print("=" * 40)
    print("  Arrow Keys = Move")
    print("  Space      = Fire")
    print("  Q / ESC    = Quit")
    print("=" * 40)

    total_reward = 0
    steps = 0
    episode = 1

    clock = pygame.time.Clock()

    running = True
    while running:
        # Process pygame events (from ALE's SDL window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        if not running:
            break

        keys = pygame.key.get_pressed()
        action = get_action_from_keys(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if reward > 0:
            print(f"  +{reward:.0f}  (Total: {total_reward:.0f})")

        if terminated or truncated:
            print(f"\n  Episode {episode} Over! Score: {total_reward:.0f} | Steps: {steps}")
            episode += 1
            total_reward = 0
            steps = 0
            obs, _ = env.reset()

        clock.tick(30)  # Cap at 30 FPS

    env.close()
    pygame.quit()
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()
