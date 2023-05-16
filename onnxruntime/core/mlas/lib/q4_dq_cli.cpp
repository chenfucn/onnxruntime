/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory. Quantized weights are expanded to fp32 before matrix
    multiplication.

--*/

#include "mlas_q4.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

char*
getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

void
usage(const char* cli)
{
    std::cout << std::endl;
    std::cout << "This utility performs int4 quantize and dequantize of a matrix, usage: " << std::endl;
    std::cout << "    " << cli << " ACTION NUM_ROWS NUM_COLS [OPTIONS]" << std::endl;
    std::cout << "    ACTION:       can be either q (quantize) or dq (de-quantize)." << std::endl;
    std::cout << "    NUM_ROWS:     number of rows in the matrix." << std::endl;
    std::cout << "    NUM_COLS:     number of columns in the matrix." << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << "    --quant_type {0, 1}." << std::endl;
    std::cout << "            Type of the block quantization." << std::endl;
    std::cout << "            0: Symmetric block quant, with fp32 scale." << std::endl;
    std::cout << "            1: (default) Block quant with fp32 scale and int8 zero-point." << std::endl;
    std::cout << "    --input_file {PATH}." << std::endl;
    std::cout << "            Path to the input file." << std::endl;
    std::cout << "    --input_offset {N}." << std::endl;
    std::cout << "            Skip the first N bytes when reading the input file." << std::endl;
    std::cout << "            Ignored when read from std in." << std::endl;
    std::cout << "    --output_file {PATH}." << std::endl;
    std::cout << "            Path to the output file. Write to std out when missing" << std::endl;
    std::cout << "    --output_format {txt,bin}" << std::endl;
    std::cout << "            txt: (default) text format: space seperated numbers." << std::endl;
    std::cout << "            bin: Binary format, can not be output to std out." << std::endl;
    std::cout << std::endl;
}


//
// Variable for commands
//
struct Cli {
    bool   dqmode = false;  // false -> quantize, true -> dequantize

    size_t num_rows = 0;
    size_t num_cols = 0;

    MLAS_BLK_QUANT_TYPE quant_type = BlkQ4Zp8;

    char*  input_file = nullptr;
    size_t input_offset = 0;

    char*  output_file = nullptr;
    bool   output_bin = false;  // false -> csv, true -> binary
};


bool
parseArgs(int argc, char* argv[], Cli& cli)
{
    if (argc < 4) {
        return false;
    }

    if (strncmp(argv[1], "q", 2) == 0) {
        cli.dqmode = false;
    } else if (strncmp(argv[1], "dq", 3) == 0) {
        cli.dqmode = true;
    } else {
        return false;
    }

    try {
        cli.num_rows = std::stoi(argv[2]);
        cli.num_cols = std::stoi(argv[3]);
    } catch (...) {
        return false;
    }

    char* quant_t = getCmdOption(argv + 4, argv + argc, "--quant_type");
    if (quant_t) {
        if (strncmp(quant_t, "0", 2) == 0) {
            cli.quant_type = BlkQ4Sym;
        }
    }

    cli.input_file = getCmdOption(argv + 4, argv + argc, "--input_file");
    char* offset_str = getCmdOption(argv + 4, argv + argc, "--input_offset");
    if (offset_str) {
        try {
            cli.input_offset = std::stoi(offset_str);
        } catch (...) {
            return false;
        }
    }

    cli.output_file = getCmdOption(argv + 4, argv + argc, "--output_file");
    char* output_format_str = getCmdOption(argv + 4, argv + argc, "--output_format");
    if (output_format_str) {
        if (strncmp(output_format_str, "csv", 4) == 0) {
            cli.output_bin = false;
        } else if (strncmp(output_format_str, "bin", 4) == 0) {
            cli.output_bin = true;
            if (!cli.output_file) {
                // can't dump binary file to std-out
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

void
readBinFile(const char* filename, size_t start, size_t expected_size, std::vector<uint8_t>& buf)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    file.seekg(start);
    fileSize -= start;
    if ((size_t)fileSize < expected_size) {
        return;
    }

    // read the data:
    buf.resize(expected_size);
    file.read((char*)buf.data(), expected_size);
}

void
writeUint8Txt(std::ostream& out, const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        out << (int)data[i] << "  ";
        if (((i+1) % 21 == 0)) {
            out << std::endl;
        }
    }
    out << std::endl;
}

void
quantize(const Cli& cli)
{
    std::vector<uint8_t> srcbuf;
    readBinFile(cli.input_file, cli.input_offset, cli.num_rows * cli.num_cols * sizeof(float), srcbuf);
    if (srcbuf.size() == 0) {
        std::cerr << "Failed to read expected amount of data from file " << cli.input_file
                  << std::endl;
        return;
    }

    size_t qsize = MlasQ4GemmPackBSize(cli.quant_type, cli.num_cols, cli.num_rows);
    std::vector<uint8_t> dstbuf(qsize);
    MlasQ4GemmPackB(cli.quant_type, dstbuf.data(), (const float*)srcbuf.data(), cli.num_cols,
                    cli.num_rows, cli.num_cols);

    if (cli.output_bin) {
        std::ofstream out(cli.output_file, std::ios::out | std::ios::binary);
        if (!out) {
            std::cerr << "Cannot open output file " << cli.output_file  << std::endl;
            return;
        }
        out.write((const char*)dstbuf.data(), dstbuf.size());
    } else {
        std::streambuf* buf;
        if (cli.output_file) {
            std::ofstream out(cli.output_file, std::ios::out);
            if (!out) {
                std::cerr << "Cannot open output file " << cli.output_file << std::endl;
                return;
            }
            buf = out.rdbuf();
        } else {
            buf = std::cout.rdbuf();
        }
        std::ostream stream(buf);
        writeUint8Txt(stream, dstbuf.data(), dstbuf.size());
    }
}

void
dequantize(const Cli& cli)
{
    size_t qsize = MlasQ4GemmPackBSize(cli.quant_type, cli.num_cols, cli.num_rows);
    std::vector<uint8_t> srcbuf;
    readBinFile(cli.input_file, cli.input_offset, qsize, srcbuf);
    if (srcbuf.size() == 0) {
        std::cerr << "Failed to read expected amount of data from file " << cli.input_file
                  << std::endl;
        return;
    }

    std::vector<float> dstbuf(cli.num_rows * cli.num_cols);
    MlasQ4GemmUnPackB(cli.quant_type, dstbuf.data(), srcbuf.data(), cli.num_cols, cli.num_rows,
                      cli.num_cols);

    if (cli.output_bin) {
        std::ofstream out(cli.output_file, std::ios::out | std::ios::binary);
        if (!out) {
            std::cerr << "Cannot open output file " << cli.output_file << std::endl;
            return;
        }
        out.write((const char*)dstbuf.data(), dstbuf.size() * sizeof(float));
    } else {
        std::streambuf* buf;
        if (cli.output_file) {
            std::ofstream out(cli.output_file, std::ios::out);
            if (!out) {
                std::cerr << "Cannot open output file " << cli.output_file << std::endl;
                return;
            }
            buf = out.rdbuf();
        } else {
            buf = std::cout.rdbuf();
        }
        std::ostream stream(buf);
        size_t lcount = 0;
        for (float v : dstbuf) {
            stream << v << "  ";
            if (++lcount >= 16) {
                stream << std::endl;
                lcount = 0;
            }
        }
        stream << std::endl;
    }
}

int
main(int argc, char* argv[])
{
    Cli cli;
    if (!parseArgs(argc, argv, cli)) {
        usage(argv[0]);
        return -1;
    }
    if (cli.dqmode) {
        dequantize(cli);
    } else {
        quantize(cli);
    }

    return 0;
}
