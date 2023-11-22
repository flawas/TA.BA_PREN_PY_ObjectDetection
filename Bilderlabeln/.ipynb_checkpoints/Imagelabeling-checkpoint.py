import os

LABELING_PATH = os.path.join('Tensorflow', 'labelimg')
if not os.path.exists(LABELING_PATH):
    !mkdir {LABELING_PATH}
if os.name == 'nt':
    !cd {LABELING_PATH} && pyrcc5 -o libs/resources.py resources.qrc
!cd {LABELING_PATH} && python labelImg.py