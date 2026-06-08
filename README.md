# EEGgame
先去emotiv官网找到cortex app，申请一个client id和client secret（需要现有一个emotiv账号），替换掉cortex-example里面sub_data.py和attention_calculator.py里的your_app_client_id和 your_app_client_secret，然后还需要有emotiv launcher这个软件来连接脑电设备或者使用虚拟设备。
然后配置一下Emotion-recognition需要的环境，我用的python版本是3.12.9
还需要分别在attention_calculator.py和real_time_vedio.py里面的# path和# parameters for loading data and images下面的代码里面改一下文件路径，按照自己电脑保存的位置改一下就行。
最终运行是运行cortex-example里的attention_calculator.py，运行后可以同时启动情绪和脑电的功能（手势识别一直加不进去，打不开，在运行这个前单独开一下吧只能），刚开始可能有点慢，要两个都打开才能输出。运行后会在桌面生成两个文件：attention.txt和emotion.txt，里面实时展示注意力和急躁指数。
# ！！！强烈建议：每次运行完后都把桌面的两个txt文件删掉再运行下一次！！！
因为以前不删掉的话桌面有同名文件就失效了，即使删掉txt文件也不会再重新生成，要想让他重新输出txt很麻烦。虽然现在我加了检查功能，理论上是不会这样了，但是我没试过有没有用，最好还是别试了，每次运行完删掉。
