#version 330 core
out vec4 FragColor;

struct Material{
    sampler2D diffuseMap;
    sampler2D specularMap;
    float shininess;
};

in vec2 TexCoord;

uniform Material material;

struct Light{
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Light light;

in vec3 Normal; 
in vec3 Pos;

uniform vec3 cameraPos;

void main()
{
    vec3 diffTex = vec3(texture(material.diffuseMap, TexCoord));
    vec3 specTex = vec3(texture(material.specularMap, TexCoord));
    vec3 ambient =  light.ambient * diffTex; 

    vec3 norm = normalize(Normal);
    vec3 diffDir = normalize(light.position - Pos);
    float diff = max(dot(norm, diffDir), 0);
    vec3 diffuse = (diff * diffTex) * light.diffuse;

    vec3 specCamDir = normalize(cameraPos - Pos);
    vec3 reflectDir = reflect(-diffDir, norm);
    float spec = pow(max(dot(reflectDir, specCamDir), 0.0), material.shininess);
    vec3 specular = (spec * specTex) * light.specular;

    vec3 result = diffuse + ambient + specular;
    FragColor = vec4(result, 1.0); 
}