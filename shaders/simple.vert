#version 430 core

layout(location = 0) in vec3 position;   // Vertex position
layout(location = 1) in vec4 color;      // Vertex color (not used)
layout(location = 2) in vec3 normal;     // Vertex normal

uniform mat4 transformationMatrix;       // Model-view-projection matrix (MVP)
uniform mat4 modelMatrix;                // Model matrix for normal transformation


out vec3 fragNormal;                     // Pass the normal to the fragment shader
out vec3 fragPos;                        // Pass the position to the fragment shader
out vec4 fragColor; 

void main()
{
    fragPos = vec3(modelMatrix * vec4(position, 1.0));   // Transform position to world space
    fragNormal = mat3(transpose(inverse(modelMatrix))) * normal;  // Transform normal
    fragColor = color;  

    gl_Position = transformationMatrix * vec4(position, 1.0);  // Transform to clip space
}
