import sys, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from matplotlib.widgets import Slider
import matplotlib.widgets as widgets

width, height = 1000, 1000  # screen size
current_circle = None


class Boids:
    """class that represents Boids simulation"""
    def __init__(self, N=50, separation=25.0, cohesion=50.0, alignment=50.0, speed=2.0):
        """initialize the Boid simulation"""
        # --Initial position and velocities--
        # self.pos = [width/2, height/2] + 10*np.random.rand(2*N).reshape(N, 2)
        self.pos = np.array([width/2, height/2]) + 10 * np.random.rand(2 * N).reshape(N, 2)
        directions = 2*math.pi*np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(directions), np.cos(directions))))
        self.N = N
        # adjustable parameters
        self.minDist = separation
        self.cohesionDist = cohesion
        self.alignmentDist = alignment
        self.maxVel = speed
        self.maxRuleVel = 0.03
        self.bird_count = N

    def tick(self, frameNum, pts, beak):
        """Update the simulation by one time step."""
        self.distMatrix = squareform(pdist(self.pos))
        self.vel += self.applyRules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.applyBC()
        pts.set_data(self.pos.reshape(2*self.N)[::2],
                     self.pos.reshape(2*self.N)[1::2])
        vec = self.pos + 10*self.vel/self.maxVel
        beak.set_data(vec.reshape(2*self.N)[::2],
                      vec.reshape(2*self.N)[1::2])

    def applyRules(self):
        # Separation
        D = self.distMatrix < self.minDist
        vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)
        # Alignment
        D = self.distMatrix < self.alignmentDist
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2
        # Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3
        return vel

    def limitVec(self, vec, maxVal):
        """limit the magnitide of the 2D vector"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag

    def limit(self, X, maxVal):
        """limit the magnitude of 2D vectors in array X to maxValue"""
        for vec in X:
            self.limitVec(vec, maxVal)

    def applyBC(self):
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR

    def buttonPress(self, event):
        """event handler for matplotlib button presses"""
        if event.inaxes == ax_button_start:
            return
        if event.inaxes == ax_button_stop:
            return
        if event.inaxes == ax_align:
            return
        if event.inaxes == ax_coh:
            return
        if event.inaxes == ax_speed:
            return
        if event.inaxes == ax_sep:
            return
        if event.button == 1:
            if event.inaxes != ax:
                return
            self.pos = np.concatenate((self.pos,
                                       np.array([[event.xdata, event.ydata]])),
                                       axis=0)
            angles = 2*math.pi*np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.N += 1
        elif event.button == 3:
            global current_circle
            if event.inaxes != ax:
                return
            radius = 50
            clicked_point = np.array([[event.xdata, event.ydata]])
            distances = np.linalg.norm(self.pos - clicked_point, axis=1)
            indices_within_radius = np.where(distances <= radius)[0]
            self.vel[indices_within_radius] += 0.1 * (self.pos[indices_within_radius] - clicked_point)
            #if current_circle:
            #    current_circle.remove()  # Видаляємо попереднє коло
            #    current_circle = None
            # Створюємо нове коло
            #current_circle = patches.Circle((event.xdata, event.ydata), radius, edgecolor='red', facecolor='none', lw=2)
            #ax.add_patch(current_circle)
            #plt.draw()
        elif event.button == 2:  # Середня кнопка миші - видалити boid
            if event.inaxes != ax:
                return
            self.removeBoid(event.xdata, event.ydata)

    def removeBoid(self, x, y):
        indices_to_remove = []
        for i in range(self.N):
            dist = np.sqrt((self.pos[i][0] - x) ** 2 + (self.pos[i][1] - y) ** 2)
            if dist < 10:  # Визначте власний радіус для видалення бід
                    indices_to_remove.append(i)
        if self.N <= 1:
            print("This is the last bird. Unable to delete!!!")
        elif indices_to_remove:
            self.pos = np.delete(self.pos, indices_to_remove, axis=0)
            self.vel = np.delete(self.vel, indices_to_remove, axis=0)
            self.N -= len(indices_to_remove)


def tick(frameNum, pts, beak, boids):
    boids.tick(frameNum, pts, beak)
    return pts, beak

def start(event):
    if event.inaxes != ax_button_start:
        return
    if event.button == 1:  # left-click
        if anim.running == False:
            anim.event_source.start()
            anim.running = True

def stop(event):
    if event.inaxes != ax_button_stop:
        return
    if event.button == 1:  # left-click
        if anim.running == True:
            anim.event_source.stop()
            anim.running = False


def update_text(frameNum, pts, beak, boids, text):
    boids.tick(frameNum, pts, beak)
    bird_count = len(boids.pos)
    text.set_text(f'Count birds: {bird_count}')
    return pts, beak, text


if __name__ == "__main__":
    boids = Boids()
    fig = plt.figure("Boids simulation")
    fig.subplots_adjust(left=0, bottom=0.17, right=1, top=1, wspace=0.2, hspace=0.2)
    ax = plt.axes(xlim=(0, width), ylim=(0, height))
    ax.axis('off')
    rectangle = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rectangle)
    pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4, c='r', marker='o', ls='None')
    text = ax.text(10, 10, f'Count birds: {boids.bird_count}', fontsize=12, color='green')

    ax_button_start = plt.axes([ 0.89, 0.09, 0.1, 0.075])
    button_start = widgets.Button(ax_button_start, 'Start')
    button_start.on_clicked(start)

    ax_button_stop = plt.axes([0.89, 0.01, 0.1, 0.075])
    button_stop = widgets.Button(ax_button_stop, 'Stop')
    button_stop.on_clicked(stop)

    # Sliders for separation, cohesion, alignment, and speed
    ax_sep = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_coh = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_align = plt.axes([0.15, 0.09, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_speed = plt.axes([0.15, 0.13, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    s_sep = Slider(ax_sep, 'Separation', 1.0, 100.0, valinit=boids.minDist)
    s_coh = Slider(ax_coh, 'Cohesion', 1.0, 100.0, valinit=boids.cohesionDist)
    s_align = Slider(ax_align, 'Alignment', 1.0, 100.0, valinit=boids.alignmentDist)
    s_speed = Slider(ax_speed, 'Speed', 0.1, 10.0, valinit=boids.maxVel)

    def update_sliders(val):
        boids.minDist = s_sep.val
        boids.cohesionDist = s_coh.val
        boids.alignmentDist = s_align.val
        boids.maxVel = s_speed.val

    s_sep.on_changed(update_sliders)
    s_coh.on_changed(update_sliders)
    s_align.on_changed(update_sliders)
    s_speed.on_changed(update_sliders)

    anim = animation.FuncAnimation(fig, update_text, fargs=(pts, beak, boids, text), interval=50)
    cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)
    anim.running = True
    plt.show()
