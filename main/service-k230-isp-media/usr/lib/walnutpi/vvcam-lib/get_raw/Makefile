# 编译器
CC = gcc

# 编译选项
CFLAGS = -Wall -Wextra -O2

# 目标文件
TARGET = get_raw

# 源文件
SRCS = main.c get_frame.c

# 对象文件
OBJS = $(SRCS:.c=.o)

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# 编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
