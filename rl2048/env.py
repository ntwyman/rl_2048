""" OpenAI gym Environments for reinforcement learning of 2048
"""
import math
import pdb
import numpy as np
import sys
import io
from gym import Env, spaces
from gym.utils import seeding
from PIL import Image, ImageDraw, ImageFont

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Text2048(Env):
    metadata  = { 'render.modes': ['human', 'ansi']}

    def __init__(self, size=4):
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 4096, (4, 4))
        self._seed()
        self._reset()
        self.done = False

    def _is_empty(self, x, y):
        return self.board[x, y] == 0

    def empty_cells(self):
        return [(x,y)
                for x in range(self.size)
                for y in range(self.size) if self._is_empty(x, y)]

    def has_move(self):
        for x in range(self.size-1):
            for y in range(self.size-1):
                val = self.board[x, y]
                if (val > 0) and (val == self.board[x+1, y] or val == self.board[x, y+1]):
                    return True
        return False

    def _fill_empty(self):
        initial_value = 4 if self.np_random.rand() > 0.9 else 2
        ec = self.empty_cells()
        x, y = ec[self.np_random.choice(len(ec))]
        self.board[x, y] = initial_value

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.board = np.zeros((self.size, self.size), np.uint16)
        self.done = False
        self._fill_empty()
        self._fill_empty()
        return self.board

    def rotate(self, times):
        self.board = np.rot90(self.board, times)

    def left(self):
        changed = False
        reward = 0

        # First pack
        for x in range(self.size):
            for y in range(self.size - 1):
                if self.board[x, y] == 0:
                    for z in range(y+1, self.size):
                        if self.board[x, z] != 0:
                            self.board[x, y] = self.board[x, z]
                            self.board[x, z] = 0
                            changed = True
                            break

        # Now merge if possible
        for x in range(self.size):
            for y in range(self.size-1):
                val = self.board[x, y]
                if val == 0:
                    continue
                if val == self.board[x, y+1]:
                    val = val * 2
                    self.board[x, y] = val
                    reward += val
                    for z in range(y+1, self.size-1):
                        self.board[x, z] = self.board[x, z + 1]
                    self.board[x, self.size - 1] = 0
                    changed = True

        if changed:
            self._fill_empty()
        return reward

    def down(self):
        self.rotate(-1)
        reward = self.left()
        self.rotate(1)
        return reward

    def up(self):
        self.rotate(1)
        reward = self.left()
        self.rotate(-1)
        return reward

    def right(self):
        self.rotate(2)
        reward = self.left()
        self.rotate(2)
        return reward


    def _step(self, action):
        if self.done:
            return self.board, 0.0, True, {}

        actions = [self.left, self.down, self.right, self.up]
        # pdb.set_trace()
        reward = actions[action]()
        done = len(self.empty_cells()) == 0 and not self.has_move()
        return self.board, reward, done, {}

    def _write_line(self, op, end_chr='|', fill_chr=' '):
        op.write(end_chr)
        for _ in range(self.size):
            op.write('{}{}'.format(fill_chr * 6, end_chr))
        op.write('\n')

    def _render(self, mode='human', close=False):
        if mode not in ['human', 'ansi']:
            return
        output = io.StringIO() if mode == 'ansi' else sys.stdout
        self._write_line(output, end_chr='+', fill_chr='-')
        for x in range(self.size):
            self._write_line(output)
            output.write('|')
            for y in range(self.size):
                output.write(' {: 4d} |'.format(self.board[x, y]))
            output.write('\n')
            self._write_line(output)
            self._write_line(output, end_chr='+', fill_chr='-')
        if mode =='ansi':
            return output


TILE_SIZE = 100
GUTTER_SIZE = 10
class Graphics2048(Text2048):
    metadata  = { 'render.modes': ['human', 'ansi','rgb_array'] }

    def __init__(self, size=4):
        self.state_size = GUTTER_SIZE + (size * (TILE_SIZE + GUTTER_SIZE))
        self.viewer = None
        super().__init__(size)
        self.observation_space = spaces.Box(0, 255, (self.state_size, self.state_size, 3))

    def _render_state(self):
        state = Image.new('RGB', (self.state_size, self.state_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(state)
        arial = ImageFont.truetype('LiberationSans-Regular.ttf', 32)
        for x in range(self.size):
            for y in range(self.size):
                # Swap x/y axes on drawing
                x0 = GUTTER_SIZE + y * (TILE_SIZE + GUTTER_SIZE)
                x1 = x0 + TILE_SIZE
                y0 = GUTTER_SIZE + x * (TILE_SIZE + GUTTER_SIZE)
                y1 = y0 + TILE_SIZE
                val = self.board[x, y]
                if val:
                    col = (192, 220, 192)
                else:
                    col = (32, 32, 32)
                draw.rectangle([x0, y0, x1, y1], fill=col)
                if val:
                    draw.text((x0+10, y0+20), str(val), font=arial, fill=(0,0,0,255))
        pixels = np.asarray(state.getdata(), np.uint8)
        self.state = np.reshape(pixels, (self.state_size, self.state_size, 3))
        return self.state

    def _reset(self):
        super()._reset()
        return self._render_state()

    def _step(self, action):
        _, reward, done, info = super()._step(action)
        return self._render_state(), reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        rendition = None
        if mode =='ansi':
            rendition = super()._render(mode, close)
        elif mode =='rgb_array':
            rendition = self.state
        else:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.state)
        return rendition

if __name__ == "__main__":
    import time
    ACTIONS =["LEFT", "DOWN", "RIGHT", "UP"]
    e = Text2048()
    for i in range(1000):
        print("\n\n\n=================================================\n")
        e._reset()
        e.render()
        done = False
        score = 0
        while not done:
            time.sleep(.1)
            action = e.action_space.sample()
            state, reward, done, _ = e._step(action)
            score += reward
            print("\n\n{:>5}, Reward {:04d} score {:05d}".format(ACTIONS[action], reward, score))
            e.render()
        print("\n----------------------- DONE -------------------------")
        time.sleep(2)
