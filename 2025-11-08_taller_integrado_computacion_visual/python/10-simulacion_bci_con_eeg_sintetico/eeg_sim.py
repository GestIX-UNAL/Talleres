# -*- coding: utf-8 -*-
# Requisitos: numpy, scipy, pygame
import numpy as np
from scipy.signal import butter, lfilter, welch
import pygame, random

# -------- Config EEG --------
FS = 256                   # Hz
WIN = 2.5                  # s por ventana
N  = int(FS*WIN)
ALPHA = (8,12)
BETA  = (13,30)
TH_ALPHA = 2.2             # umbral relativo simple
TH_BETA  = 2.0

# -------- Síntesis ----------
def synth_eeg(n, fs, a_amp=1.0, b_amp=0.8, noise=0.4):
    t = np.arange(n)/fs
    alpha = a_amp*np.sin(2*np.pi*10*t + np.random.rand()*2*np.pi)
    beta  = b_amp*np.sin(2*np.pi*20*t + np.random.rand()*2*np.pi)
    pink  = noise*np.cumsum(np.random.randn(n)); pink /= np.max(np.abs(pink)+1e-6)
    return alpha + beta + 0.4*pink

def bandpass(sig, lo, hi, fs, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return lfilter(b, a, sig)

def rel_power(sig, fs):
    f, P = welch(sig, fs=fs, nperseg=fs//2)
    total = P.mean()
    return total

# -------- Visual ----------
pygame.init()
W,H = 900, 600
screen = pygame.display.set_mode((W,H))
pygame.display.set_caption("EEG sintético → Alpha/Beta → Control visual")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Estado visual
scale = 1.0        # controlado por Alpha (relajación)
rot = 0.0
spin = 1.0         # velocidad base
color = (80,160,255)
burst_timer = 0    # explosión de partículas por Beta

particles = []
def spawn_burst(center, n=70):
    for _ in range(n):
        a = np.random.uniform(0, 2*np.pi)
        v = np.random.uniform(2, 6)
        particles.append([center[0], center[1], v*np.cos(a), v*np.sin(a), 30])

def update_particles():
    alive = []
    for p in particles:
        p[0] += p[2]; p[1] += p[3]; p[4] -= 1
        if p[4] > 0: alive.append(p)
    return alive

base_size = 160
center = (W//2, H//2)

def draw_bar(x,y,w,h,val,maxv,label,col):
    v = max(0.0, min(1.0, val/maxv))
    pygame.draw.rect(screen, (60,60,70), (x,y,w,h), 1)
    pygame.draw.rect(screen, col, (x,y,int(w*v),h))
    screen.blit(font.render(f"{label}: {val:.2f}", True, (210,210,210)), (x, y-22))

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); raise SystemExit

    # ---- EEG ----
    sig = synth_eeg(N, FS, a_amp=np.random.uniform(0.9,1.4),
                             b_amp=np.random.uniform(0.6,1.1),
                             noise=0.35)
    a_sig = bandpass(sig, *ALPHA, FS)
    b_sig = bandpass(sig, *BETA,  FS)

    total = rel_power(sig, FS)
    a_rel = rel_power(a_sig, FS)/max(1e-9, total)
    b_rel = rel_power(b_sig, FS)/max(1e-9, total)

    # Mapeos:
    # Alpha alta → más grande y rotación más lenta
    target_scale = 1.0 + min(1.0, 0.6*(a_rel-1.0))
    scale = 0.9*scale + 0.1*target_scale
    spin = max(0.2, 1.2 - 0.5*(a_rel-1.0))

    # Beta alta → estallido de partículas
    if b_rel > TH_BETA:
        spawn_burst(center, n=80)
        burst_timer = 12

    # ---- Dibujo ----
    screen.fill((15,15,20))
    # Partículas
    particles = update_particles()
    for p in particles:
        alpha = max(0, min(255, int(255*(p[4]/30.0))))
        pygame.draw.circle(screen, (255,200,120,alpha), (int(p[0]), int(p[1])), 3)

    # Figura principal
    rot += spin
    size = int(base_size*scale)
    surf = pygame.Surface((size,size), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0,0,size,size), border_radius=16)
    pygame.draw.rect(surf, (255,255,255,40), (3,3,size-6,size-6), width=3, border_radius=14)
    rsurf = pygame.transform.rotozoom(surf, rot, 1.0)
    screen.blit(rsurf, rsurf.get_rect(center=center))

    # HUD EEG
    draw_bar(40, 520, 300, 24, a_rel, 4.0, "Alpha 8–12Hz (rel.)", (120,200,255))
    draw_bar(380,520, 300, 24, b_rel, 4.0, "Beta 13–30Hz (rel.)", (255,180,120))
    thr = font.render(f"TH_BETA={TH_BETA}  |  Alpha↑ ⇒ más grande + giro lento  |  Beta>TH ⇒ partículas", True, (180,180,180))
    screen.blit(thr, (40, 552))

    pygame.display.flip()
    clock.tick(12)
