#include <stdio.h>
#include "get_frame.h"

int main(int argc, char *argv[])
{
    printf("开始获取摄像头帧数据...\n");
    
    int ret = get_frame();
    
    if (ret == 0) {
        printf("帧数据获取成功！\n");
    } else {
        printf("帧数据获取失败，错误码: %d\n", ret);
        return -1;
    }
    
    return 0;
}
