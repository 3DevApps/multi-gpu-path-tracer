
#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>

#define CheckedGLCall(x) do { printOpenGLErrors(">>BEFORE<< "#x, __FILE__, __LINE__); (x); printOpenGLErrors(#x, __FILE__, __LINE__); } while (0)
#define CheckedGLResult(x) (x); printOpenGLErrors(#x, __FILE__, __LINE__);
#define CheckExistingErrors(x) printOpenGLErrors(">>BEFORE<< "#x, __FILE__, __LINE__);

void printOpenGLErrors(char const * const function, char const * const file, int const line) {
	bool succeeded = true;
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		char const *errorString = (char const *) gluErrorString(error);
            throw std::runtime_error(
                "OpenGL Error in " + std::string(file) + 
                " at line" + std::to_string(line) + 
                " calling function " + std::string(function) + 
                ": " + std::string(errorString));
	}
}

void printShaderInfoLog(GLint const shader) {
	int infoLogLength = 0;
	int charsWritten = 0;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (infoLogLength > 0) {
		GLchar * infoLog = new GLchar[infoLogLength];
		glGetShaderInfoLog(shader, infoLogLength, &charsWritten, infoLog);
		std::cout << "Shader Info Log:" << std::endl << infoLog << std::endl;
		delete [] infoLog;
	}
}