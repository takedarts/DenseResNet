This is a dataset directory of TinyImageNet.

[step 1] Download `tiny-imagenet-200.zip`.
https://tiny-imagenet.herokuapp.com/

[step 2] Save these files in a dataset directory.
% tree data
data
└── tinyimagenet
    ├── readme.txt
    └── tiny-imagenet-200.zip

[step 3] Run a preparation script: `src/prepare.py`.
% python src/prepare.py tinyimagenet

