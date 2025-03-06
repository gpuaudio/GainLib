# Gain Processor Client Library
This project illustrates how to write a simple client library for
a GPU processor. The client library takes care of loading the GPU Audio
engine and the gain processor and offers a lightweight interface to process
audio data either synchronously or asynchronuosly with double buffering.

## GainInterface
Interface for the client library and public include for projects using the library.

## GPUCreate
Function to create an instance of the client library, i.e., GPUGainProcessor.

## GPUGainProcessor
Actual implementation of the GainInterface.

# GPUGainProcessorTests
Unit tests to check correctness of processor and library and to illustrate how to use the
library to process data.
