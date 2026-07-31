CC      = gcc
CFLAGS  = -Wall -Wextra -g -O2
TARGET  = v4l2-cap
SOURCES = main.c v4l2_cap.c save.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
