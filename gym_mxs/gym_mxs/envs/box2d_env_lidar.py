from . import box2d_env

class MxsEnvBox2DLidar(box2d_env.MxsEnvBox2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, use_lidar=True)
      