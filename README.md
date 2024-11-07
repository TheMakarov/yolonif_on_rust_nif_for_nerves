# YOLOv5 DNN NIF Implementation for Elixir

## Overview

This project integrates the YOLOv5 model, written in Rust, with Elixir using Rustler. The main goal is to provide a seamless way to run YOLOv5 object detection directly from Elixir applications. The Rust code is compiled as a NIF (Native Implemented Function), allowing Elixir to call the YOLOv5 model efficiently.

## Prerequisites

- Elixir (`asdf` or `exenv` recommended)
- Rust (`rustup` recommended)
- Rustler (Rust library for creating Elixir NIFs)
- YOLOv5 model and dependencies

## Setup

### Elixir Application

1. Clone the repository:
    ```sh
    git clone https://github.com/TheMakarov/yolonif_on_opencv_dnn.git
    cd yolonif_on_opencv_dnn
    ```

2. Install Elixir dependencies:
    ```sh
    mix deps.get
    ```

3. Set up the Rustler NIF:
    ```sh
    mix rustler.install
    ```

### Rust Application

1. Navigate to the `native` directory:
    ```sh
    cd native
    ```

2. Build the Rust project:
    ```sh
    cargo build --release
    ```

3. Return to the project root and compile the NIF:
    ```sh
    mix compile
    ```

### Running the Application

1. Start the Elixir application:
    ```sh
    iex -S mix
    ```

2. Use the YOLOv5 NIF in your Elixir code:
    ```elixir
    SomeElixirModule.start_detect()
    ```

## Features

- **YOLOv5 Integration**: Run YOLOv5 object detection directly from Elixir applications.
- **Efficient Performance**: Utilize the power of Rust for high-performance object detection.
- **Seamless Integration**: Leverage Rustler to create a smooth integration between Rust and Elixir.

## How It Works

- The Rust project implements the YOLOv5 model and exposes it as a NIF.
- The Elixir application uses Rustler to call the YOLOv5 NIF, allowing for efficient object detection.
- The YOLOv5 model processes images and returns detection results, which can be further processed in Elixir.
- For each time the func `read_chunk` is called , the rust green thread , saves the the Mat data into a Jpeg format in the cross sharable object `ResourceArc`, i.e from rust to elixir , then u can save it or show it from elixir .

## Note

Please be aware that this project contains some bad words in the code. We apologize for any offense caused and will work on cleaning up the language in future updates.

## Acknowledgments

- Thanks to the Elixir and Rust communities for their excellent documentation and support.
- Thanks to the YOLOv5 team for their groundbreaking work in object detection.
