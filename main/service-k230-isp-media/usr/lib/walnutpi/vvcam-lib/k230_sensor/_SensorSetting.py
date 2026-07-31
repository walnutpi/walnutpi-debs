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
        '''
        传感器设置类
        :param i2c_addr: I2C地址
        :param sensor: 传感器名称
        :param mode: 传感器模式列表
        '''
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
    
    def get_mode(self, isp: int) -> SensorMode | None:
        '''
        返回当前isp处于SensorMode列表中的哪个模式
        :param isp: ISP编号 (0, 1, 2)
        :return: 匹配的SensorMode，如果未找到返回None
        '''
        try:
            with open(DEV_ISP, 'r') as fp:
                content = fp.read()
        except Exception as e:
            print(f"Failed to read {DEV_ISP}: {e}")
            return None

        # 解析对应 isp port 的 mode 值
        target_port = f"isp0 port{isp}:"
        mode_value = None
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if target_port in line:
                # 在接下来的几行中查找 mode
                for j in range(i + 1, min(i + 10, len(lines))):
                    if 'mode' in lines[j]:
                        # 提取 mode 数值，格式: "mode     : 0"
                        try:
                            mode_value = int(lines[j].split(':')[1].strip())
                        except (IndexError, ValueError):
                            pass
                        break
                break

        if mode_value is None:
            print(f"Mode not found for isp{isp} in {DEV_ISP}")
            return None

        # 在 mode_list 中查找匹配的 SensorMode
        for m in self.mode_list:
            if m.mode == mode_value:
                return m
            # 输出m.mode
            print(f"{mode_value} m.mode: {m.mode} ")

        print(f"Mode {mode_value} not found in mode_list for isp{isp}")
        return None
    def set_closest_mode(self, isp: int, width: int, height: int, fps: int = 90) -> None:
        '''
        设置为最接近的模式
        参数:
            isp: ISP编号
            width: 宽度
            height: 高度
            fps: 帧率
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
            self.set_mode(isp, self.mode_list[0])
        else :
            self.set_mode(isp, best_mode)
    

    def set_mode(self, isp: int ,mode: SensorMode):
        '''
        设置传感器模式并写入ISP配置
        :param mode: SensorMode对象，包含宽度、高度、帧率和模式编号
        '''
        print(f"Setting sensor to mode {mode.mode}")
        return
        try:
            # 写入sensor名称
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"{isp} sensor={self.sensor}\n")
            
            # 写入mode
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"{isp} mode={mode.mode}\n")
            
            # 写入xml路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"{isp} xml=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}.xml\n")
            
            # 写入manual json路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"{isp} manu_json=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}_manual.json\n")
            
            # 写入auto json路径
            with open(DEV_ISP, 'w') as fp:
                fp.write(f"{isp} auto_json=/etc/vvcam/{self.sensor}-{mode.width}x{mode.height}_auto.json\n")
            
            print(f"ISP settings written for sensor {self.sensor}, mode {mode.mode}")
        except Exception as e:
            print(f"Failed to write ISP settings: {e}")
