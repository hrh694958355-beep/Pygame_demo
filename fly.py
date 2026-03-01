import pygame
import random
import math

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('飞机大战')

# 资源加载容错处理（没有图片/音效时用默认图形替代）
try:
    icon = pygame.image.load('ufo.png')
    pygame.display.set_icon(icon)
except:
    # 没有图标时用默认Surface替代
    icon = pygame.Surface((32, 32))
    icon.fill((255, 0, 0))
    pygame.display.set_icon(icon)

try:
    bgImg = pygame.image.load('bg.png')
except:
    bgImg = pygame.Surface((800, 600))
    bgImg.fill((0, 0, 0))  # 黑色背景

# 音效加载容错
try:
    pygame.mixer.music.load('bg.wav')
    pygame.mixer.music.play(-1)  # 单曲循环
except:
    pass  # 没有背景音乐则跳过

try:
    bao_sound = pygame.mixer.Sound('exp.wav')
except:
    bao_sound = None  # 没有击中音效则设为None

# 玩家飞机（容错处理）
try:
    playerImg = pygame.image.load('player.png')
except:
    playerImg = pygame.Surface((64, 64))
    playerImg.fill((0, 255, 0))  # 绿色方块替代玩家飞机

playerX = 400  # 玩家的X坐标
playerY = 500  # 玩家的Y坐标
playerStep = 0  # 玩家移动的速度
player_width = 64  # 玩家飞机宽度（用于碰撞检测）
player_height = 64  # 玩家飞机高度

# 分数
score = 0
font = pygame.font.Font('freesansbold.ttf', 32)
if not font:
    font = pygame.font.SysFont(None, 32)  # 备用字体

def show_score():
    text = f"Score: {score}"
    score_render = font.render(text, True, (0,255,0))
    screen.blit(score_render, (10,10))

# 游戏结束
is_over = False
over_font = pygame.font.Font('freesansbold.ttf', 64)
if not over_font:
    over_font = pygame.font.SysFont(None, 64)

def check_is_over():
    if is_over:
        text = "Game Over"
        render = over_font.render(text, True, (255,0,0))
        screen.blit(render, (200,250))

# 敌人类（修复重置逻辑）
class Enemy():
    def __init__(self):
        try:
            self.img = pygame.image.load('enemy.png')
        except:
            self.img = pygame.Surface((64, 64))
            self.img.fill((255, 0, 0))  # 红色方块替代敌人
        self.x = random.randint(0, 736)  # 修正随机范围（避免出界）
        self.y = random.randint(50, 250)
        self.step = random.randint(1, 2)  # 敌人移动速度（1-2）
        self.width = 64
        self.height = 64

    # 当被射中时，恢复位置
    def reset(self):
        self.x = random.randint(0, 736)
        self.y = random.randint(50, 200)
        self.step = random.randint(1, 3)

# 初始化敌人
number_of_enemies = 6
enemies = []
for i in range(number_of_enemies):
    enemies.append(Enemy())

# 两个点之间的距离（碰撞检测辅助）
def distance(bx, by, ex, ey):
    a = bx - ex
    b = by - ey
    return math.sqrt(a*a + b*b)

# 子弹类（修复遍历移除异常）
class Bullet():
    def __init__(self):
        try:
            self.img = pygame.image.load('bullet.png')
        except:
            self.img = pygame.Surface((16, 32))
            self.img.fill((255, 255, 0))  # 黄色方块替代子弹
        self.x = playerX + 24  # 修正子弹位置（居中）
        self.y = playerY - 16
        self.step = 10
        self.width = 16
        self.height = 32

    # 击中检测
    def hit(self):
        global score
        hit_enemy = None
        for e in enemies:
            if distance(self.x, self.y, e.x, e.y) < 30:
                hit_enemy = e
                break
        if hit_enemy:
            if bao_sound:
                bao_sound.play()
            hit_enemy.reset()
            score += 1
            return True  # 击中返回True
        return False

# 保存子弹的列表
bullets = []

# 显示并移动子弹（修复遍历移除问题）
def show_bullets():
    # 先收集需要移除的子弹
    bullets_to_remove = []
    for b in bullets:
        screen.blit(b.img, (b.x, b.y))
        # 移动子弹
        b.y -= b.step
        # 击中检测
        if b.hit():
            bullets_to_remove.append(b)
        # 子弹出界
        if b.y < 0:
            bullets_to_remove.append(b)
    # 批量移除子弹（避免遍历中修改列表）
    for b in bullets_to_remove:
        if b in bullets:
            bullets.remove(b)

# 显示敌人+移动+碰撞检测（核心修复：玩家与敌人碰撞即失败）
def show_enemy():
    global is_over
    if is_over:
        return
    for e in enemies:
        screen.blit(e.img,(e.x, e.y))
        # 敌人左右移动
        e.x += e.step
        if e.x > 736 or e.x < 0:
            e.step *= -1
            e.y += 40  # 触边下移
        # 玩家与敌人碰撞检测（矩形碰撞，更精准）
        if (playerX < e.x + e.width and
            playerX + player_width > e.x and
            playerY < e.y + e.height and
            playerY + player_height > e.y):
            is_over = True
            print("游戏结束：撞到敌人了！")
        # 敌人下移过多也结束
        if e.y > 450:
            is_over = True
            print("游戏结束：敌人突破防线！")

# 玩家移动（边界限制）
def move_player():
    global playerX
    playerX += playerStep
    if playerX > 736:
        playerX = 736
    if playerX < 0:
        playerX = 0

# 重置游戏（用于AI训练时重置）
def reset_game():
    global score, is_over, playerX, playerY, enemies, bullets
    score = 0
    is_over = False
    playerX = 400
    playerY = 500
    bullets = []
    enemies = []
    for i in range(number_of_enemies):
        enemies.append(Enemy())

# 游戏主循环
running = True
while running:
    screen.blit(bgImg,(0,0))
    show_score()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # 手动控制（仅玩家模式）
        if not is_over:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    playerStep = 5
                elif event.key == pygame.K_LEFT:
                    playerStep = -5
                elif event.key == pygame.K_SPACE:
                    bullets.append(Bullet())
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_RIGHT, pygame.K_LEFT]:
                    playerStep = 0
        # 按R重置游戏
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            reset_game()

    if not is_over:
        screen.blit(playerImg, (playerX, playerY))
        move_player()
        show_enemy()
        show_bullets()
    else:
        check_is_over()

    pygame.display.update()

pygame.quit()