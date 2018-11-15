import os
import cs236605.jupyter_utils as jupyter_utils

THIS_FILE = os.path.abspath(__file__)
NB_DIR = os.path.join(os.path.dirname(THIS_FILE), '..')


def run_nb(name):
    nb_path = os.path.join(NB_DIR, name)
    jupyter_utils.nbconvert(nb_path, execute=True, debug=True, stdout=True)


def test_part1():
    run_nb('Part1_Datasets.ipynb')


def test_part2():
    run_nb('Part2_KNN.ipynb')


def test_part3():
    run_nb('Part3_LinearClassifier.ipynb')


def test_part4():
    run_nb('Part4_LinearRegression.ipynb')
