#version 430 core

in vec3 fragNormal;   // The normal vector from the vertex shader
in vec3 fragPos;      // The fragment position in world space
in vec4 fragColor;  

uniform vec3 viewPos;      // Position of the camera/viewer
uniform vec3 lightColor;   
 
// Output color to the framebuffer
out vec4 outputColor;

void main()
{
    vec3 lightPos = vec3(0.0, 60.0, 40.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0); // White light
    // Normalize the normal vector
    vec3 norm = normalize(fragNormal);

    // Compute the light direction
    vec3 lightDir = normalize(lightPos - fragPos);

    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Combine ambient and diffuse lighting
    outputColor = vec4((ambient + diffuse), 1) * fragColor;
    //outputColor = vec4(norm.x,norm.y,norm.z,1);
}
