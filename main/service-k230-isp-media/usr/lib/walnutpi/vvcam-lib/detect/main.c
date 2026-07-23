#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

#include "get_frame.h"

#define DEV_ISP "/proc/vsi/isp_subdev0"
#define DEV_I2C "/dev/i2c-0"

struct sensor_setting
{
    uint16_t i2c_addr;
    const char *sensor;
    const char *mode;
    const char *xml;
    const char *manu_json;
    const char *auto_json;
};

struct sensor_setting gc2093 = {
    .i2c_addr = 0x37,
    .sensor = "gc2093",
    .mode = "0",
    .xml = "/etc/vvcam/gc2093-1920x1080.xml",
    .manu_json = "/etc/vvcam/gc2093-1920x1080_manual.json",
    .auto_json = "/etc/vvcam/gc2093-1920x1080_auto.json"};
struct sensor_setting ov5647 = {
    .i2c_addr = 0x36,
    .sensor = "ov5647",
    .mode = "0",
    .xml = "/etc/vvcam/ov5647.xml",
    .manu_json = "/etc/vvcam/ov5647.manual.json",
    .auto_json = "/etc/vvcam/ov5647.auto.json"};

struct sensor_setting *sensorlist[] = {&gc2093, &ov5647,
                                       NULL};

// 打开文件路径 DEV_ISP , 写入指定的配置参数，完成isp的配置
void write_isp_setting(struct sensor_setting *setting)
{
    FILE *fp;

    fp = fopen(DEV_ISP, "w");
    if (fp == NULL)
    {
        printf("Failed to open %s\n", DEV_ISP);
        return;
    }
    fprintf(fp, "0 sensor=%s\n", setting->sensor);
    fclose(fp);

    fp = fopen(DEV_ISP, "w");
    fprintf(fp, "0 mode=%s\n", setting->mode);
    fclose(fp);

    fp = fopen(DEV_ISP, "w");
    fprintf(fp, "0 xml=%s\n", setting->xml);
    fclose(fp);

    fp = fopen(DEV_ISP, "w");
    fprintf(fp, "0 manu_json=%s\n", setting->manu_json);
    fclose(fp);

    fp = fopen(DEV_ISP, "w");
    fprintf(fp, "0 auto_json=%s\n", setting->auto_json);
    fclose(fp);
}

// 传入一个i2c地址，判断该地址是否在i2c总线上
bool is_addr_in_i2c(uint16_t addr)
{
    int fd;
    uint8_t addr_buf[1];

    fd = open(DEV_I2C, O_RDWR);
    if (fd < 0)
    {
        printf("Failed to open %s\n", DEV_I2C);
        return false;
    }

    if (ioctl(fd, I2C_SLAVE_FORCE, addr) < 0)
    {
        close(fd);
        return false;
    }

    // 尝试读取数据来检测设备是否存在
    // 发送一个简单的读取命令来测试地址
    if (read(fd, addr_buf, 1) < 0)
    {
        close(fd);
        return false;
    }

    close(fd);
    return true;
}

int main(int argc, char *argv[])
{


    // get_frame();

    //   遍历sensorlist，判断每个sensor的i2c地址是否存在，如果存在则调用write_isp_setting函数配置isp
    for (int i = 0; sensorlist[i] != NULL; i++)
    {
        if (is_addr_in_i2c(sensorlist[i]->i2c_addr))
        {
            printf("Sensor %s found at i2c address 0x%x\n", sensorlist[i]->sensor, sensorlist[i]->i2c_addr);
            write_isp_setting(sensorlist[i]);
            return 0;
        }
    }
    write_isp_setting(&gc2093);

    return 0;
}