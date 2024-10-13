Reinforcement Learning Agent for 2048
=

WIP - @nick

This is very much a WIP. This is where I am sticking my work on trying to write a Reinforcement Learning agent which can learn to play 2048.

This will use [gym](https://github.com/openai/gym) as the environment.

The 2048 environment is in env.py atm. There is a Text and graphical version. The idea is to first build an agent which can play a text representation of the game, then a second agent which will be able to take an RGB image which shows the screen as its environment.

I am using a series of medium articles on [Reinforcement Learning](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) to help understand what RL is and how it differs from simpler deep learning with tensorflow.

The code as I follow along with the articles is in `rlwtf/` with names that match the articles, e.g. part1.py is the code corresponding with the article titled [Part 1 - Two-armed Bandit](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149).


INSTALLATION
==

To get a suitable python installed using pyenv, I needed the following:

```> PYTHON_CONFIGURE_OPTS="--enable-framework" CFLAGS="-I$(brew --prefix openssl)/include" LDFLAGS="-L$(brew --prefix openssl)/lib" pyenv install 3.6.2
> pyenv shell 3.6.2
> pip install wheel tensorflow numpy gym matplotlib```
