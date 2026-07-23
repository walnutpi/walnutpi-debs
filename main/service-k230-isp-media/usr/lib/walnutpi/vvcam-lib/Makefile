CC = gcc
CFLAGS = -fPIC -shared -g -O0
SRCDIR = src
SOURCEDIR = $(SRCDIR)
INCDIR = include
TARGET = libvvcam.so
SOURCES = $(wildcard $(SOURCEDIR)/*.c)
OBJECTS = $(SOURCES:.c=.o)

# 新增 detect 相关变量
DETECT_DIR = detect
DETECT_SOURCES = $(wildcard $(DETECT_DIR)/*.c)
DETECT_OBJECTS = $(DETECT_SOURCES:.c=.o)
DETECT_TARGET = detect_sensor

all: $(TARGET) $(DETECT_TARGET)

install: install-lib install-detect install-configs

install-lib: $(TARGET)
	install -m 755 $(TARGET) /usr/lib/
	install -m 644 $(INCDIR)/*.h /usr/include/

install-detect: $(DETECT_TARGET)
	install -m 755 $(DETECT_TARGET) /usr/bin/

install-configs:
	mkdir -p /etc/vvcam
	cp -r configs/* /etc/vvcam/

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

$(DETECT_TARGET): $(DETECT_OBJECTS)
	$(CC) -o $@ $^ -I$(INCDIR)

%.o: %.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(DETECT_DIR)/%.o: $(DETECT_DIR)/%.c
	$(CC) $(filter-out -fPIC -shared, $(CFLAGS)) -I$(INCDIR) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(DETECT_OBJECTS) $(TARGET) $(DETECT_TARGET)

.PHONY: all clean install install-lib install-detect install-configs

$(OBJECTS) $(DETECT_OBJECTS): | $(INCDIR)