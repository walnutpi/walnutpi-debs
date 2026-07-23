import fcntl

DEV_ISP = "/proc/vsi/isp_subdev0"

class SensorMode:
    def __init__(self, width: int, height: int, fps: int, mode: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode

class SensorSetting:
    mode_list: list[SensorMode] = []
    def __init__(self, i2c_addr: int, sensor: str, mode: list[SensorMode] = []):
        self.i2c_addr = i2c_addr
        self.sensor = sensor
        self.mode_list = mode

    def check_i2c(self, i2c_bus: int = 0) -> bool:
        '''
        检查在指定的I2C总线上是否存在本传感器
        :param i2c_bus: I2C总线编号，默认为0（对应/dev/i2c-0）
        :return: 如果传感器存在返回True，否则返回False
        '''
        I2C_DEVICE = f"/dev/i2c-{i2c_bus}"
        I2C_SLAVE_FORCE = 0x0706
        
        try:
            # 打开I2C设备
            fd = open(I2C_DEVICE, 'r+b', buffering=0)
        except Exception as e:
            print(f"Failed to open {I2C_DEVICE}: {e}")
            return False
        
        try:
            # 设置I2C从设备地址
            fcntl.ioctl(fd, I2C_SLAVE_FORCE, self.i2c_addr)
            
            # 尝试读取一个字节来检测设备是否存在
            data = fd.read(1)
            
            if len(data) > 0:
                # 设备响应，说明找到了
                print(f"Sensor {self.sensor} found at I2C-{i2c_bus} address 0x{self.i2c_addr:02x}")
                return True
            else:
                return False
                
        except Exception as e:
            # 该地址没有设备响应
            return False
        finally:
            # 关闭I2C设备
            fd.close()
    
    def get_mode(self, width: int, height: int, fps: int) -> SensorMode:
        '''
        传入宽度、高度、帧率，返回最接近的SensorMode对象
        优先选择宽高大于传入参数的模式
        '''
        if not self.mode_list:
            return None
        
        best_mode = None
        min_diff = float('inf')
        
        for m in self.mode_list:
            if m.width >= width and m.height >= height:
                diff = abs(m.width - width) + abs(m.height - height) + abs(m.fps - fps)
                
                if diff < min_diff:
                    min_diff = diff
                    best_mode = m
        
        if best_mode is None:
            min_diff = float('inf')
            for m in self.mode_list:
                diff = abs(m.width - width) + abs(m.height - height) + abs(m.fps - fps)
                if diff < min_diff:
                    min_diff = diff
                    best_mode = m
            return self.mode_list[0]
        else :
            return best_mode
    def set_mode(self, mode: SensorMode):
        '''
        设置传感器模式并写入ISP配置
        :param mode: SensorMode对象，包含宽度、高度、帧率和模式编号
        '''
        print(f"Setting sensor to mode {mode.mode}")
        try:
            # 写入sensor名称
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"0 sensor={self.sensor}\n")
            
            # 写入mode
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"0 mode={mode.mode}\n")
            
            # 写入xml路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"0 xml=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}.xml\n")
            
            # 写入manual json路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"0 manu_json=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}_manual.json\n")
            
            # 写入auto json路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"0 auto_json=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}_auto.json\n")
            
            print(f"ISP settings written for sensor {self.sensor}, mode {mode.mode}")
        except Exception as e:
            print(f"Failed to write ISP settings: {e}")
