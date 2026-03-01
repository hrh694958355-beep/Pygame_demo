import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # 经验回放池

# ========== 1. 基础配置（核心：关闭窗口渲染） ==========
# 设备选择（优先GPU，没有则CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# 游戏初始化（仅创建内存画布，不显示窗口）
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
# 关键：创建内存画布，替代窗口渲染
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))  # 不显示窗口

# 游戏参数
number_of_enemies = 6
shoot_cooldown = 0
SHOOT_COOLDOWN_MAX = 10
max_steps_per_episode = 800  # 减少每轮步数，提速
max_episodes = 5000           # 总训练轮数

# 单个敌人配置
ENEMY_ROW_DOWN = 30  # 单个敌人碰边后下移距离
ENEMY_BORDER_MARGIN = 5  # 边界边距
ENEMY_BASE_SPEED = 2  # 敌人基础移动速度

# ========== 2. 轻量化DQN网络（提速核心） ==========
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 轻量化卷积层：适配40×30的画面
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)  # 40×30 → 19×14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # 19×14 → 8×6
        # 轻量化全连接层
        self.fc1 = nn.Linear(32 * 8 * 6, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)  # 展平
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 初始化网络、优化器、损失函数
model = DQN().to(DEVICE)
target_model = DQN().to(DEVICE)  # 目标网络（提升稳定性）
target_model.load_state_dict(model.state_dict())  # 同步参数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# ========== 3. 经验回放池（解决数据相关性） ==========
REPLAY_BUFFER_SIZE = 100000  # 经验池大小
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
BATCH_SIZE = 32  # 每次训练批次

# ========== 4. 游戏核心类/变量 ==========
class Enemy():
    def __init__(self):
        self.img = pygame.Surface((64, 64))
        self.img.fill((255, 0, 0))
        # 初始位置（避免贴边）
        self.x = random.randint(ENEMY_BORDER_MARGIN, SCREEN_WIDTH-64-ENEMY_BORDER_MARGIN)
        self.y = random.randint(50, 255)
        self.width = 64
        self.height = 64
        # 单个敌人独立属性
        self.step = random.randint(ENEMY_BASE_SPEED, ENEMY_BASE_SPEED+2)  # 独立速度
        self.direction_x = random.choice([-1, 1])  # 独立初始方向
        self.down_cooldown = 0  # 下移冷却（防止连续下移）

    def update(self):
        """单个敌人独立移动逻辑：碰边仅自己下移+反向"""
        # 下移冷却：刚下移后短暂不检测边界，避免反复触发
        if self.down_cooldown > 0:
            self.down_cooldown -= 1

        # 水平移动
        self.x += self.direction_x * self.step

        # 单个敌人边界检测（仅自己触发）
        hit_border = False
        # 右边界
        if self.direction_x == 1 and self.x + self.width >= SCREEN_WIDTH - ENEMY_BORDER_MARGIN:
            hit_border = True
        # 左边界
        elif self.direction_x == -1 and self.x <= ENEMY_BORDER_MARGIN:
            hit_border = True

        # 仅当前敌人碰边时，执行自己的下移+反向
        if hit_border and self.down_cooldown == 0:
            self.direction_x *= -1  # 自己反向
            self.y += ENEMY_ROW_DOWN  # 自己下移
            self.down_cooldown = 5  # 冷却5帧，避免刚下移又碰边
            # 修正位置，避免卡边界
            self.x = max(ENEMY_BORDER_MARGIN, min(self.x, SCREEN_WIDTH - self.width - ENEMY_BORDER_MARGIN))

    def reset(self):
        """重置单个敌人状态"""
        self.x = random.randint(ENEMY_BORDER_MARGIN, SCREEN_WIDTH-64-ENEMY_BORDER_MARGIN)
        self.y = random.randint(50, 200)
        self.step = random.randint(ENEMY_BASE_SPEED, ENEMY_BASE_SPEED+2)
        self.direction_x = random.choice([-1, 1])
        self.down_cooldown = 0

class Bullet():
    def __init__(self, playerX, playerY):
        self.img = pygame.Surface((16, 32))
        self.img.fill((255, 255, 0))
        self.x = playerX + 24
        self.y = playerY - 16
        self.step = 6
        self.width = 16
        self.height = 32

    def hit(self, enemies):
        for e in enemies:
            if (self.x < e.x + e.width and self.x + self.width > e.x and
                self.y < e.y + e.height and self.y + self.height > e.y):
                e.reset()
                return True
        return False

# 玩家变量
playerX = SCREEN_WIDTH//2
playerY = SCREEN_HEIGHT - 64
score = 0
is_over = False
current_steps = 0

# ========== 5. 轻量化画面预处理（极速版） ==========
def preprocess_screen():
    """仅在内存中处理画面，无渲染"""
    # 1. 将内存画布转为numpy数组
    screen_array = pygame.surfarray.array3d(screen)
    # 2. 转为灰度图（简化计算）
    gray_screen = np.dot(screen_array[..., :3], [0.299, 0.587, 0.114])
    # 3. 缩放为40×30（大幅降低计算量）
    gray_screen = pygame.transform.scale(pygame.surfarray.make_surface(gray_screen), (40, 30))
    gray_screen = pygame.surfarray.array2d(gray_screen)
    # 4. 归一化+维度调整（适配模型输入）
    gray_screen = gray_screen / 255.0
    gray_screen = np.expand_dims(gray_screen, axis=0)  # batch维度
    gray_screen = np.expand_dims(gray_screen, axis=0)  # channel维度
    # 5. 转为torch张量
    return torch.tensor(gray_screen, dtype=torch.float32).to(DEVICE)

# ========== 6. 强化学习核心函数 ==========
# ε-贪心策略（动态衰减）
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 0.0005  # 每步衰减
epsilon = EPSILON_START

def choose_action(state):
    global epsilon
    # 衰减ε
    epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)
    # 探索：随机动作
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])  # 0=左移，1=右移，2=射击
    # 利用：模型预测最优动作
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()

# 奖励函数
def get_reward():
    reward = 0
    if is_over:
        reward = -200  # 失败大惩罚
    else:
        # 存活奖励
        reward += 0.1
        # 击中奖励
        reward += score * 1
        # 原地惩罚
        if abs(playerX - SCREEN_WIDTH//2) < 50:
            reward -= 0.5
        # 靠近敌人惩罚
        closest_enemy = min(enemies, key=lambda e: math.hypot(playerX-e.x, playerY-e.y))
        if math.hypot(playerX - closest_enemy.x, playerY - closest_enemy.y) < 100:
            reward -= 1
        # 边界惩罚
        if playerX < 50 or playerX > SCREEN_WIDTH-50:
            reward -= 0.5
    return reward

# 训练模型（从经验池采样）
def train_model():
    if len(replay_buffer) < BATCH_SIZE:
        return  # 经验池不足，不训练
    # 随机采样批次经验
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    # 转为张量
    states = torch.cat(states).to(DEVICE)
    next_states = torch.cat(next_states).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.int64).to(DEVICE)

    # 计算当前Q值
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # 计算目标Q值（用目标网络）
    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * 0.95 * next_q  # 折扣因子0.95

    # 计算损失并更新
    loss = loss_fn(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ========== 7. 游戏重置函数 ==========
def reset_game():
    global playerX, playerY, score, is_over, current_steps, shoot_cooldown, enemies, bullets
    playerX = SCREEN_WIDTH//2
    playerY = SCREEN_HEIGHT - 64
    score = 0
    is_over = False
    current_steps = 0
    shoot_cooldown = 0
    enemies = [Enemy() for _ in range(number_of_enemies)]
    bullets = []

# ========== 8. 极速训练主循环（无任何画面渲染） ==========
reset_game()
episode = 0
train_loss = 0
episode_scores = []
print("开始无渲染训练...")

while episode < max_episodes:
    # 1. 清空内存画布（替代窗口清屏）
    screen.fill((0, 0, 0))

    # 2. 处理退出事件（仅终端Ctrl+C退出）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # 3. 游戏未结束时执行AI逻辑
    if not is_over:
        current_steps += 1
        # 3.1 获取当前画面状态
        state = preprocess_screen()
        # 3.2 选择动作
        action = choose_action(state)
        # 3.3 执行动作
        if action == 0:  # 左移
            playerX -= 5
        elif action == 1:  # 右移
            playerX += 5
        elif action == 2 and shoot_cooldown == 0:  # 射击（冷却）
            bullets.append(Bullet(playerX, playerY))
            shoot_cooldown = SHOOT_COOLDOWN_MAX

        # 边界限制
        playerX = max(0, min(playerX, SCREEN_WIDTH-64))
        # 更新射击冷却
        if shoot_cooldown > 0:
            shoot_cooldown -= 1

        # 3.4 更新游戏对象
        # 敌人更新（单个独立更新）
        for e in enemies:
            e.update()
            # 碰撞检测
            if (playerX < e.x + e.width and playerX + 64 > e.x and
                playerY < e.y + e.height and playerY + 64 > e.y):
                is_over = True
        # 子弹更新
        bullets_to_remove = []
        for b in bullets:
            screen.blit(b.img, (b.x, b.y))  # 画到内存画布，不显示
            b.y -= b.step
            # 击中检测
            if b.hit(enemies):
                score += 1
                bullets_to_remove.append(b)
            # 子弹出界
            if b.y < 0:
                bullets_to_remove.append(b)
        # 移除子弹
        for b in bullets_to_remove:
            if b in bullets:
                bullets.remove(b)

        # 3.5 绘制游戏对象到内存画布（不显示）
        # 绘制玩家
        player_img = pygame.Surface((64, 64))
        player_img.fill((0, 255, 0))
        screen.blit(player_img, (playerX, playerY))
        # 绘制敌人
        for e in enemies:
            screen.blit(e.img, (e.x, e.y))

        # 3.6 获取奖励和下一状态
        reward = get_reward()
        next_state = preprocess_screen()
        done = is_over or (current_steps >= max_steps_per_episode)
        if done:
            is_over = True

        # 3.7 存储经验到回放池
        replay_buffer.append((state, action, reward, next_state, done))
        # 3.8 每4步训练一次（大幅提速）
        if current_steps % 4 == 0:
            loss = train_model()
            if loss:
                train_loss = loss

    # 4. 游戏结束/步数用尽，重置并记录
    if is_over or current_steps >= max_steps_per_episode:
        episode += 1
        episode_scores.append(score)
        # 每10轮打印日志
        if episode % 10 == 0:
            avg_score = np.mean(episode_scores[-10:])
            print(f"第{episode}轮 | 近10轮平均得分：{avg_score:.1f} | ε：{epsilon:.3f} | 损失：{train_loss:.4f}")
        # 每50轮同步目标网络+保存模型
        if episode % 50 == 0:
            target_model.load_state_dict(model.state_dict())
            # 保存模型
            torch.save(model.state_dict(), f"dqn_airplane_ep{episode}.pth")
            print(f"✅ 已保存模型（第{episode}轮），同步目标网络")
        # 重置游戏
        reset_game()

# 保存最终模型
torch.save(model.state_dict(), "dqn_airplane_final.pth")
print("\n训练完成！最终模型已保存为 dqn_airplane_final.pth")
pygame.quit()