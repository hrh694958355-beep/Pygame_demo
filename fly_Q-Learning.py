import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # 经验回放池/Experience replay buffer

# ========== 1. 基础配置（核心：关闭窗口渲染） ==========
# Basic Configuration (Core: Disable window rendering)
# 设备选择（优先GPU，没有则CPU）
# Device selection (GPU first, CPU if not available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE} / Using device: {DEVICE}")

# 游戏初始化（仅创建内存画布，不显示窗口）
# Game initialization (create memory canvas only, no window display)
pygame.init()
SCREEN_WIDTH = 800    # 屏幕宽度/Screen width
SCREEN_HEIGHT = 600   # 屏幕高度/Screen height
# 关键：创建内存画布，替代窗口渲染
# Key: Create memory canvas to replace window rendering
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))  # 不显示窗口/No window display

# 游戏参数/Game parameters
number_of_enemies = 6          # 敌人数量/Number of enemies
shoot_cooldown = 0             # 射击冷却计数/Shoot cooldown counter
SHOOT_COOLDOWN_MAX = 10        # 最大射击冷却帧数/Max shoot cooldown frames
max_steps_per_episode = 800    # 每轮最大步数（减少步数提速）/Max steps per episode (reduce for speed)
max_episodes = 5000            # 总训练轮数/Total training episodes

# 单个敌人配置/Single enemy configuration
ENEMY_ROW_DOWN = 30            # 单个敌人碰边后下移距离/Down distance when enemy hits border
ENEMY_BORDER_MARGIN = 5        # 边界边距/Border margin
ENEMY_BASE_SPEED = 2           # 敌人基础移动速度/Enemy base movement speed

# ========== 2. 轻量化DQN网络（提速核心） ==========
# Lightweight DQN Network (Core for speed up)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 轻量化卷积层：适配40×30的画面
        # Lightweight convolutional layers: adapt to 40×30 screen
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)  # 40×30 → 19×14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # 19×14 → 8×6
        # 轻量化全连接层/Lightweight fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 6, 256)
        self.fc2 = nn.Linear(256, 3)  # 3个动作：左移/右移/射击/3 actions: left/right/shoot

    def forward(self, x):
        """前向传播/Forward propagation"""
        x = torch.relu(self.conv1(x))   # 激活函数/Activation function
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)                # 展平特征图/Flatten feature maps
        x = torch.relu(self.fc1(x))
        return self.fc2(x)              # 输出Q值/Output Q-values

# 初始化网络、优化器、损失函数
# Initialize network, optimizer, loss function
model = DQN().to(DEVICE)                          # 主网络/Main network
target_model = DQN().to(DEVICE)                   # 目标网络（提升稳定性）/Target network (improve stability)
target_model.load_state_dict(model.state_dict())  # 同步参数/Sync parameters
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 优化器/Optimizer
loss_fn = nn.MSELoss()                                 # 损失函数/Loss function

# ========== 3. 经验回放池（解决数据相关性） ==========
# Experience Replay Buffer (Solve data correlation)
REPLAY_BUFFER_SIZE = 100000  # 经验池大小/Replay buffer size
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)  # 双端队列存储经验/Deque to store experiences
BATCH_SIZE = 32              # 每次训练批次/Batch size for training

# ========== 4. 游戏核心类/变量 ==========
# Game Core Classes/Variables
class Enemy():
    """敌人类/Enemy class"""
    def __init__(self):
        self.img = pygame.Surface((64, 64))  # 敌人图像/Enemy image
        self.img.fill((255, 0, 0))           # 红色/Red color
        # 初始位置（避免贴边）/Initial position (avoid border)
        self.x = random.randint(ENEMY_BORDER_MARGIN, SCREEN_WIDTH-64-ENEMY_BORDER_MARGIN)
        self.y = random.randint(50, 255)
        self.width = 64   # 宽度/Width
        self.height = 64  # 高度/Height
        # 单个敌人独立属性/Independent attributes for single enemy
        self.step = random.randint(ENEMY_BASE_SPEED, ENEMY_BASE_SPEED+2)  # 独立速度/Independent speed
        self.direction_x = random.choice([-1, 1])  # 独立初始方向/Independent initial direction
        self.down_cooldown = 0  # 下移冷却（防止连续下移）/Down cooldown (prevent continuous down)

    def update(self):
        """单个敌人独立移动逻辑：碰边仅自己下移+反向
        Independent movement logic for single enemy: only self move down and reverse when hit border"""
        # 下移冷却：刚下移后短暂不检测边界，避免反复触发
        # Down cooldown: no border detection shortly after moving down to avoid repeated trigger
        if self.down_cooldown > 0:
            self.down_cooldown -= 1

        # 水平移动/Horizontal movement
        self.x += self.direction_x * self.step

        # 单个敌人边界检测（仅自己触发）/Single enemy border detection (only self trigger)
        hit_border = False
        # 右边界/Right border
        if self.direction_x == 1 and self.x + self.width >= SCREEN_WIDTH - ENEMY_BORDER_MARGIN:
            hit_border = True
        # 左边界/Left border
        elif self.direction_x == -1 and self.x <= ENEMY_BORDER_MARGIN:
            hit_border = True

        # 仅当前敌人碰边时，执行自己的下移+反向
        # Only current enemy move down and reverse when hit border
        if hit_border and self.down_cooldown == 0:
            self.direction_x *= -1  # 自己反向/Self reverse direction
            self.y += ENEMY_ROW_DOWN  # 自己下移/Self move down
            self.down_cooldown = 5   # 冷却5帧，避免刚下移又碰边/Cooldown 5 frames
            # 修正位置，避免卡边界/Correct position to avoid stuck at border
            self.x = max(ENEMY_BORDER_MARGIN, min(self.x, SCREEN_WIDTH - self.width - ENEMY_BORDER_MARGIN))

    def reset(self):
        """重置单个敌人状态/Reset single enemy state"""
        self.x = random.randint(ENEMY_BORDER_MARGIN, SCREEN_WIDTH-64-ENEMY_BORDER_MARGIN)
        self.y = random.randint(50, 200)
        self.step = random.randint(ENEMY_BASE_SPEED, ENEMY_BASE_SPEED+2)
        self.direction_x = random.choice([-1, 1])
        self.down_cooldown = 0

class Bullet():
    """子弹类/Bullet class"""
    def __init__(self, playerX, playerY):
        self.img = pygame.Surface((16, 32))  # 子弹图像/Bullet image
        self.img.fill((255, 255, 0))         # 黄色/Yellow color
        self.x = playerX + 24                # 初始X坐标/Initial X coordinate
        self.y = playerY - 16                # 初始Y坐标/Initial Y coordinate
        self.step = 6                        # 移动速度/Movement speed
        self.width = 16                      # 宽度/Width
        self.height = 32                     # 高度/Height

    def hit(self, enemies):
        """检测是否击中敌人/Detect if hit enemy"""
        for e in enemies:
            if (self.x < e.x + e.width and self.x + self.width > e.x and
                self.y < e.y + e.height and self.y + self.height > e.y):
                e.reset()  # 重置敌人位置/Reset enemy position
                return True
        return False

# 玩家变量/Player variables
playerX = SCREEN_WIDTH//2    # 玩家X坐标/Player X coordinate
playerY = SCREEN_HEIGHT - 64 # 玩家Y坐标/Player Y coordinate
score = 0                    # 得分/Score
is_over = False              # 游戏结束标志/Game over flag
current_steps = 0            # 当前步数/Current steps

# ========== 5. 轻量化画面预处理（极速版） ==========
# Lightweight Screen Preprocessing (Ultra-fast version)
def preprocess_screen():
    """仅在内存中处理画面，无渲染/Process screen in memory only, no rendering"""
    # 1. 将内存画布转为numpy数组/Convert memory canvas to numpy array
    screen_array = pygame.surfarray.array3d(screen)
    # 2. 转为灰度图（简化计算）/Convert to grayscale (simplify calculation)
    gray_screen = np.dot(screen_array[..., :3], [0.299, 0.587, 0.114])
    # 3. 缩放为40×30（大幅降低计算量）/Resize to 40×30 (reduce computation significantly)
    gray_screen = pygame.transform.scale(pygame.surfarray.make_surface(gray_screen), (40, 30))
    gray_screen = pygame.surfarray.array2d(gray_screen)
    # 4. 归一化+维度调整（适配模型输入）/Normalization + dimension adjustment (adapt to model input)
    gray_screen = gray_screen / 255.0  # 归一化到0-1/Normalize to 0-1
    gray_screen = np.expand_dims(gray_screen, axis=0)  # batch维度/Batch dimension
    gray_screen = np.expand_dims(gray_screen, axis=0)  # channel维度/Channel dimension
    # 5. 转为torch张量/Convert to torch tensor
    return torch.tensor(gray_screen, dtype=torch.float32).to(DEVICE)

# ========== 6. 强化学习核心函数 ==========
# Reinforcement Learning Core Functions
# ε-贪心策略（动态衰减）/ε-Greedy Strategy (Dynamic decay)
EPSILON_START = 0.9    # 初始探索概率/Initial exploration probability
EPSILON_END = 0.05     # 最终探索概率/Final exploration probability
EPSILON_DECAY = 0.0005 # 每步衰减率/Decay rate per step
epsilon = EPSILON_START

def choose_action(state):
    """选择动作（ε-贪心）/Choose action (ε-Greedy)"""
    global epsilon
    # 衰减ε/Decay epsilon
    epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)
    # 探索：随机动作/Exploration: random action
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])  # 0=左移/left, 1=右移/right, 2=射击/shoot
    # 利用：模型预测最优动作/Exploitation: model predict optimal action
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()

def get_reward():
    """奖励函数（塑造合理的奖励机制）/Reward function (shape reasonable reward mechanism)"""
    reward = 0
    if is_over:
        reward = -200  # 失败大惩罚/Large penalty for failure
    else:
        # 存活奖励/Survival reward
        reward += 0.1
        # 击中奖励/Hit reward
        reward += score * 1
        # 原地惩罚/Penalty for staying in place
        if abs(playerX - SCREEN_WIDTH//2) < 50:
            reward -= 0.5
        # 靠近敌人惩罚/Penalty for approaching enemy
        closest_enemy = min(enemies, key=lambda e: math.hypot(playerX-e.x, playerY-e.y))
        if math.hypot(playerX - closest_enemy.x, playerY - closest_enemy.y) < 100:
            reward -= 1
        # 边界惩罚/Penalty for approaching border
        if playerX < 50 or playerX > SCREEN_WIDTH-50:
            reward -= 0.5
    return reward

def train_model():
    """训练模型（从经验池采样）/Train model (sample from replay buffer)"""
    if len(replay_buffer) < BATCH_SIZE:
        return  # 经验池不足，不训练/Not enough experience, skip training
    # 随机采样批次经验/Random sample batch of experiences
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    # 转为张量/Convert to tensors
    states = torch.cat(states).to(DEVICE)
    next_states = torch.cat(next_states).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64).to(DEVICE)

    # 计算当前Q值/Calculate current Q-values
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # 计算目标Q值（用目标网络）/Calculate target Q-values (use target network)
    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * 0.95 * next_q  # 折扣因子0.95/Discount factor 0.95

    # 计算损失并更新/Calculate loss and update model
    loss = loss_fn(current_q, target_q)
    optimizer.zero_grad()  # 清空梯度/Clear gradients
    loss.backward()        # 反向传播/Backpropagation
    optimizer.step()       # 更新参数/Update parameters
    return loss.item()

# ========== 7. 游戏重置函数 ==========
# Game Reset Function
def reset_game():
    """重置游戏状态/Reset game state"""
    global playerX, playerY, score, is_over, current_steps, shoot_cooldown, enemies, bullets
    playerX = SCREEN_WIDTH//2
    playerY = SCREEN_HEIGHT - 64
    score = 0
    is_over = False
    current_steps = 0
    shoot_cooldown = 0
    enemies = [Enemy() for _ in range(number_of_enemies)]  # 重新创建敌人/Recreate enemies
    bullets = []                                           # 清空子弹/Clear bullets

# ========== 8. 极速训练主循环（无任何画面渲染） ==========
# Ultra-fast Training Main Loop (No screen rendering at all)
reset_game()
episode = 0          # 当前训练轮数/Current training episode
train_loss = 0       # 训练损失/Training loss
episode_scores = []  # 每轮得分/Score per episode
print("开始无渲染训练... / Start non-rendering training...")

while episode < max_episodes:
    # 1. 清空内存画布（替代窗口清屏）/Clear memory canvas (replace window clear)
    screen.fill((0, 0, 0))

    # 2. 处理退出事件（仅终端Ctrl+C退出）/Handle exit event (only Ctrl+C in terminal)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # 3. 游戏未结束时执行AI逻辑/Execute AI logic when game not over
    if not is_over:
        current_steps += 1
        # 3.1 获取当前画面状态/Get current screen state
        state = preprocess_screen()
        # 3.2 选择动作/Choose action
        action = choose_action(state)
        # 3.3 执行动作/Execute action
        if action == 0:  # 左移/Move left
            playerX -= 5
        elif action == 1:  # 右移/Move right
            playerX += 5
        elif action == 2 and shoot_cooldown == 0:  # 射击（冷却）/Shoot (cooldown)
            bullets.append(Bullet(playerX, playerY))
            shoot_cooldown = SHOOT_COOLDOWN_MAX

        # 边界限制/Border limit
        playerX = max(0, min(playerX, SCREEN_WIDTH-64))
        # 更新射击冷却/Update shoot cooldown
        if shoot_cooldown > 0:
            shoot_cooldown -= 1

        # 3.4 更新游戏对象/Update game objects
        # 敌人更新（单个独立更新）/Enemy update (independent update)
        for e in enemies:
            e.update()
            # 碰撞检测/Collision detection
            if (playerX < e.x + e.width and playerX + 64 > e.x and
                playerY < e.y + e.height and playerY + 64 > e.y):
                is_over = True  # 碰撞则游戏结束/Game over if collision
        # 子弹更新/Bullet update
        bullets_to_remove = []
        for b in bullets:
            screen.blit(b.img, (b.x, b.y))  # 画到内存画布，不显示/Draw to memory canvas (no display)
            b.y -= b.step                   # 子弹上移/Bullet move up
            # 击中检测/Hit detection
            if b.hit(enemies):
                score += 1                  # 得分+1/Score +1
                bullets_to_remove.append(b) # 标记移除子弹/Mark bullet for removal
            # 子弹出界/Bullet out of screen
            if b.y < 0:
                bullets_to_remove.append(b) # 标记移除子弹/Mark bullet for removal
        # 移除子弹/Remove bullets
        for b in bullets_to_remove:
            if b in bullets:
                bullets.remove(b)

        # 3.5 绘制游戏对象到内存画布（不显示）/Draw game objects to memory canvas (no display)
        # 绘制玩家/Draw player
        player_img = pygame.Surface((64, 64))  # 玩家图像/Player image
        player_img.fill((0, 255, 0))           # 绿色/Green color
        screen.blit(player_img, (playerX, playerY))
        # 绘制敌人/Draw enemies
        for e in enemies:
            screen.blit(e.img, (e.x, e.y))

        # 3.6 获取奖励和下一状态/Get reward and next state
        reward = get_reward()
        next_state = preprocess_screen()
        done = is_over or (current_steps >= max_steps_per_episode)
        if done:
            is_over = True

        # 3.7 存储经验到回放池/Store experience to replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        # 3.8 每4步训练一次（大幅提速）/Train every 4 steps (speed up significantly)
        if current_steps % 4 == 0:
            loss = train_model()
            if loss:
                train_loss = loss

    # 4. 游戏结束/步数用尽，重置并记录/Reset and record when game over/steps exhausted
    if is_over or current_steps >= max_steps_per_episode:
        episode += 1
        episode_scores.append(score)
        # 每10轮打印日志/Print log every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(episode_scores[-10:])
            print(f"第{episode}轮 | 近10轮平均得分：{avg_score:.1f} | ε：{epsilon:.3f} | 损失：{train_loss:.4f}")
            print(f"Episode {episode} | Avg score (last 10): {avg_score:.1f} | ε: {epsilon:.3f} | Loss: {train_loss:.4f}")
        # 每50轮同步目标网络+保存模型/Sync target network + save model every 50 episodes
        if episode % 50 == 0:
            target_model.load_state_dict(model.state_dict())
            # 保存模型/Save model
            torch.save(model.state_dict(), f"dqn_airplane_ep{episode}.pth")
            print(f"✅ 已保存模型（第{episode}轮），同步目标网络")
            print(f"✅ Model saved (Episode {episode}), target network synced")
        # 重置游戏/Reset game
        reset_game()

# 保存最终模型/Save final model
torch.save(model.state_dict(), "dqn_airplane_final.pth")
print("\n训练完成！最终模型已保存为 dqn_airplane_final.pth")
print("\nTraining completed! Final model saved as dqn_airplane_final.pth")
pygame.quit()