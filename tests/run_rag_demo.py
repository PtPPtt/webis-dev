import sys, os
sys.path.insert(0, os.path.abspath('legacy_v1'))
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))

print('PYTHONPATH:', sys.path[:3])

try:
    from RAG.rag_agent_demo import main
    main()
    print('\nDEMO_FINISHED')
except Exception as e:
    print('DEMO_ERROR:', e)
    import traceback
    traceback.print_exc()
