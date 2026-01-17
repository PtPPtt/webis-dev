import pkgutil
import webis.plugins.processors

# 获取 webis.plugins.processors 包下的所有模块
modules = list(pkgutil.iter_modules(webis.plugins.processors.__path__))

print(f"在 'webis.plugins.processors' 中找到的子模块数量: {len(modules)}")
print("子模块列表:")
for _, name, _ in modules:
    print(f"  - {name}")