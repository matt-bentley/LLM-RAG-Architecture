# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy project file and restore dependencies
COPY ["Ai.Rag.Demo.csproj", "."]
RUN dotnet restore "Ai.Rag.Demo.csproj"

# Copy all source files
COPY . .

# Build and publish the application
RUN dotnet publish "Ai.Rag.Demo.csproj" -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app

# Copy published application
COPY --from=build /app/publish .

ENTRYPOINT ["dotnet", "Ai.Rag.Demo.dll"]
