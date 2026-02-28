#pragma once

#include "Modules/ModuleManager.h"

class FMARLPluginModule : public IModuleInterface
{
public:

	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
